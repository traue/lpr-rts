"""Tipos de dados compartilhados por todos os estagios do pipeline.

Todas as estruturas sao ``dataclasses`` simples e serializaveis: o pipeline as
troca entre threads e o modulo de analytics as grava no banco sem conversao
manual.

Duas convencoes que os dados gravados dependem de serem respeitadas:

* **Coordenadas.** Dentro do pipeline, as caixas ``(x1, y1, x2, y2)`` estao em
  pixels da imagem *processada* (que pode ter sido reduzida por
  ``source.resize_width``). ``Frame.scale`` guarda a razao processada/original,
  e tudo o que vai para o banco e convertido de volta para pixels da imagem
  *original* - o mesmo espaco do ground truth e da calibracao.
* **Tempo.** ``Frame.source_pts`` e o tempo de midia (segundos desde o inicio do
  video, ou desde a abertura do stream ao vivo). E a base de tempo de toda
  medida fisica: velocidade, permanencia, trajetoria. ``Frame.timestamp`` e o
  relogio de processamento (``time.monotonic()``) e serve apenas para medir
  latencia - usa-lo como base fisica faria a velocidade depender da maquina.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .taxonomy import ObjectCategory, ObjectClass, category, has_plate, is_vulnerable

BBox = tuple[float, float, float, float]

# Teto de pontos de trajetoria mantidos em memoria por trilha. Acima dele a
# trajetoria e dizimada (um ponto a cada dois), nunca truncada no inicio: um
# veiculo parado por horas nao pode crescer sem limite num sistema 24/7, mas
# perder o comeco da trajetoria falsearia permanencia e sentido. A trajetoria
# completa, quadro a quadro, vai para a tabela ``track_points`` do banco.
MAX_TRAJECTORY_POINTS = 2000


class PlateLayout(str, Enum):
    """Layout sintatico da placa brasileira."""

    MERCOSUL = "mercosul"      # LLLNLNN  (ex.: ABC1D23)
    LEGACY = "legacy"          # LLLNNNN  (ex.: ABC1234)
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Frame:
    """Um quadro capturado, com os dois relogios e a escala de coordenadas."""

    index: int
    image: np.ndarray
    timestamp: float           # time.monotonic() quando o quadro ficou disponivel (latencia)
    source_pts: float = 0.0    # tempo de midia em segundos (base fisica dos dados)
    width: int = 0             # dimensoes da imagem processada
    height: int = 0
    scale: float = 1.0         # largura processada / largura original
    orig_width: int = 0        # dimensoes da imagem original
    orig_height: int = 0
    decode_ms: float = 0.0     # custo de leitura/decodificacao (e redimensionamento)
    # Quadro ORIGINAL, quando a fonte foi reduzida (resize_width) e o OCR le na
    # resolucao da camera (ocr.full_resolution). So uma referencia: o quadro ja
    # foi decodificado; descarta-lo e o que custava os pixels das placas pequenas.
    full_image: np.ndarray | None = None
    # Rotacao (codigo do cv2.rotate) que leva ``full_image`` a orientacao de
    # ``image``: o video vertical de celular e guardado cru, deitado.
    full_rotation: int | None = None

    def __post_init__(self) -> None:
        if self.height == 0 or self.width == 0:
            self.height, self.width = self.image.shape[:2]
        if self.scale <= 0:
            self.scale = 1.0
        if self.orig_width == 0 or self.orig_height == 0:
            self.orig_width = int(round(self.width / self.scale))
            self.orig_height = int(round(self.height / self.scale))


@dataclass(slots=True)
class FrameRef:
    """Referencia leve a um quadro ja processado (indice e tempo de midia).

    Usada quando uma decisao acontece fora do laco de um quadro - por exemplo,
    ao encerrar as trilhas no fim do video -, para gravar o instante correto
    em vez de um zero.
    """

    index: int
    source_pts: float
    scale: float = 1.0


@dataclass(slots=True)
class Detection:
    """Deteccao bruta de um detector (objeto ou placa)."""

    bbox: BBox
    score: float
    label: str = "object"
    class_id: int = -1
    # Trilha a que a deteccao foi associada neste quadro (None se nenhuma).
    # Preenchido pelo rastreador e pela associacao placa->veiculo; permite
    # gravar no banco qual deteccao alimentou qual trilha.
    track_id: int | None = None

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class PlateRead:
    """Uma tentativa de OCR em um unico quadro (hipotese, ainda nao fundida).

    Tentativas que nao produziram texto tambem sao representadas (``text`` vazio):
    excluir as falhas do denominador superestimaria a exatidao do motor.
    """

    text: str                              # texto normalizado e corrigido pela gramatica
    confidence: float                      # confianca agregada do motor (escala propria de cada motor)
    char_confidences: list[float] = field(default_factory=list)
    layout: PlateLayout = PlateLayout.UNKNOWN
    valid_syntax: bool = False
    bbox: BBox | None = None               # caixa da placa (coordenadas processadas)
    frame_index: int = -1                  # quadro do qual o recorte foi tirado
    video_time_s: float = 0.0              # tempo de midia desse quadro
    engine: str = ""
    ocr_ms: float = 0.0                    # tempo dentro do motor
    wait_ms: float = 0.0                   # espera na fila do pool antes de comecar
    collected_frame: int = -1              # quadro em que o resultado entrou na fusao
    sharpness: float = 0.0                 # variancia do Laplaciano do recorte
    brightness: float = 0.0                # media de luminancia do recorte (0-255)
    contrast: float = 0.0                  # desvio-padrao de luminancia do recorte
    raw_text: str = ""                     # saida do motor ANTES da gramatica
    n_corrections: int = 0                 # caracteres trocados pela mascara brasileira


@dataclass(slots=True)
class TrackState:
    """Estado de um objeto rastreado ao longo do tempo.

    Tempos (``first_seen``, ``last_seen``, ``timestamps``) estao em tempo de
    midia (segundos), nunca no relogio de processamento.
    """

    track_id: int
    bbox: BBox
    score: float
    object_class: ObjectClass = ObjectClass.DESCONHECIDO
    age: int = 0                  # quadros (da fonte) desde a criacao
    hits: int = 0                 # quadros com associacao a uma deteccao
    time_since_update: int = 0    # quadros (da fonte) desde a ultima associacao
    first_seen: float = 0.0       # tempo de midia da primeira deteccao (s)
    last_seen: float = 0.0        # tempo de midia da ultima deteccao (s)
    first_frame: int = -1
    last_frame: int = -1
    # Disponibilidade do primeiro quadro no host; separado do tempo de midia.
    # Usado para atraso real de decisao, inclusive OCR que termina apos EOF.
    first_available_monotonic: float | None = None
    # Confirmada = atingiu min_hits. Trilhas nao confirmadas sao hipoteses do
    # rastreador e nunca sao entregues ao resto do pipeline.
    confirmed: bool = False
    # True a partir do primeiro quadro em que a trilha foi entregue ao pipeline.
    # So trilhas reportadas sao gravadas no banco ao terminar.
    reported: bool = False
    # Historico de pontos medidos (centro da caixa, ponto de contato com o solo,
    # tempo de midia e quadro), dizimado acima de MAX_TRAJECTORY_POINTS.
    centroids: list[tuple[float, float]] = field(default_factory=list)
    ground_points: list[tuple[float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    # Leituras de OCR acumuladas (fusao multi-frame). Pode ser limitado por
    # fusion.max_reads_per_track; n_reads_total conta todas.
    reads: list[PlateRead] = field(default_factory=list)
    n_reads_total: int = 0
    # Melhor estimativa corrente da placa (provisoria ate ser publicada).
    plate_text: str | None = None
    plate_confidence: float = 0.0
    plate_layout: PlateLayout = PlateLayout.UNKNOWN
    # True a partir do momento em que a fusao atingiu os criterios de publicacao.
    # A interface distingue leitura confirmada de palpite provisorio por ele.
    plate_stable: bool = False
    # Decisao efetivamente entregue (o evento gravado), se houve. Com
    # fusion.publish_policy = "revise", published_text e o texto VIGENTE (a ultima
    # revisao); quadro e instante sao os da PRIMEIRA decisao.
    published_text: str | None = None
    published_frame: int = -1
    published_time_s: float | None = None
    first_published_text: str | None = None
    published_decision: str = ""  # "stable" | "revision" | "exit_relaxed" (a vigente)
    n_revisions: int = 0
    # Analytics de trafego
    crossed_lines: set[str] = field(default_factory=set)
    speed_kmh: float | None = None
    direction: str | None = None
    best_crop: np.ndarray | None = None
    best_crop_score: float = -1.0
    end_reason: str = ""          # "lost" | "end_of_stream" | "reset" (vazio enquanto viva)

    @property
    def category(self) -> ObjectCategory:
        """Categoria semantica (pessoa, veiculo motorizado, animal...)."""
        return category(self.object_class)

    @property
    def can_have_plate(self) -> bool:
        """True se vale gastar OCR nesta trilha."""
        return has_plate(self.object_class)

    @property
    def is_vulnerable(self) -> bool:
        """Usuario vulneravel da via (pedestre, ciclista, motociclista)."""
        return is_vulnerable(self.object_class)

    @property
    def dwell_s(self) -> float:
        """Permanencia em cena, em tempo de midia."""
        return max(0.0, self.last_seen - self.first_seen)

    def add_point(self, frame_index: int, t: float) -> None:
        """Registra a caixa atual como ponto medido da trajetoria."""
        x1, y1, x2, y2 = self.bbox
        self.centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        self.ground_points.append(((x1 + x2) / 2.0, y2))
        self.timestamps.append(float(t))
        self.frames.append(int(frame_index))
        if len(self.frames) > MAX_TRAJECTORY_POINTS:
            self._decimate()

    def _decimate(self) -> None:
        """Mantem um ponto a cada dois, preservando o primeiro e o ultimo."""
        ultimo = len(self.frames) - 1
        manter = list(range(0, ultimo, 2)) + [ultimo]
        self.centroids = [self.centroids[i] for i in manter]
        self.ground_points = [self.ground_points[i] for i in manter]
        self.timestamps = [self.timestamps[i] for i in manter]
        self.frames = [self.frames[i] for i in manter]

    def snapshot(self) -> TrackState:
        """Copia segura para outra thread (a interface desenha enquanto o pipeline altera)."""
        c = copy.copy(self)
        c.centroids = list(self.centroids)
        c.ground_points = list(self.ground_points)
        c.timestamps = list(self.timestamps)
        c.frames = list(self.frames)
        c.reads = list(self.reads)
        c.crossed_lines = set(self.crossed_lines)
        return c


@dataclass(slots=True)
class StageTiming:
    """Tempos de um quadro, em milissegundos.

    ``total_ms`` e o tempo de *servico* (soma dos estagios). ``latency_ms`` e a
    latencia ponta a ponta: do instante em que o quadro ficou disponivel ate o
    resultado sair do pipeline - inclui a espera em fila (``queue_ms``), que e
    justamente o que cresce quando o sistema nao acompanha a fonte.
    """

    capture_ms: float = 0.0       # leitura/decodificacao do quadro
    queue_ms: float = 0.0         # espera entre a captura e o inicio do processamento
    detect_ms: float = 0.0
    track_ms: float = 0.0
    ocr_ms: float = 0.0           # offline: OCR sincrono; tempo real: agendar + colher
    analytics_ms: float = 0.0
    total_ms: float = 0.0         # tempo de servico
    latency_ms: float = 0.0       # ponta a ponta (fila + servico)

    def as_dict(self) -> dict[str, float]:
        return {
            "capture_ms": self.capture_ms,
            "queue_ms": self.queue_ms,
            "detect_ms": self.detect_ms,
            "track_ms": self.track_ms,
            "ocr_ms": self.ocr_ms,
            "analytics_ms": self.analytics_ms,
            "total_ms": self.total_ms,
            "latency_ms": self.latency_ms,
        }


@dataclass(slots=True)
class FrameResult:
    """Saida completa do pipeline para um quadro: consumida pela UI e pelo store."""

    frame: Frame
    vehicles: list[Detection] = field(default_factory=list)
    plates: list[Detection] = field(default_factory=list)
    tracks: list[TrackState] = field(default_factory=list)
    reads: list[PlateRead] = field(default_factory=list)
    timing: StageTiming = field(default_factory=StageTiming)
    dropped_frames: int = 0       # descartes acumulados desde o inicio da sessao
    frames_skipped: int = 0       # quadros da fonte pulados desde o quadro processado anterior
    detection_ran: bool = True    # False nos quadros intermediarios de detect_every_n
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def counts_by_category(self) -> dict[str, int]:
        """Quantas trilhas ativas por categoria semantica neste quadro."""
        counts: dict[str, int] = {}
        for track in self.tracks:
            key = track.category.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for track in self.tracks:
            key = track.object_class.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def n_vehicles(self) -> int:
        return sum(
            1 for t in self.tracks if t.category is ObjectCategory.VEICULO_MOTORIZADO
        )

    @property
    def n_persons(self) -> int:
        return sum(1 for t in self.tracks if t.category is ObjectCategory.PESSOA)

    @property
    def n_vulnerable(self) -> int:
        """Pedestres, ciclistas e motociclistas - recorte de seguranca viaria."""
        return sum(1 for t in self.tracks if t.is_vulnerable)

    def snapshot(self) -> FrameResult:
        """Copia com trilhas copiadas: segura para entregar a interface grafica."""
        c = copy.copy(self)
        c.tracks = [t.snapshot() for t in self.tracks]
        c.vehicles = list(self.vehicles)
        c.plates = list(self.plates)
        c.reads = list(self.reads)
        return c
