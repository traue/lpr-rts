"""Configuracao do sistema: dataclasses tipadas com carga/gravacao em YAML.

Toda a config e serializada para dentro do banco em cada sessao. Isso e o que
torna um experimento reproduzivel meses depois: o resultado carrega consigo os
parametros exatos que o produziram.

Chaves desconhecidas no YAML sao **erro**, nao silencio: um ``min_read: 5``
digitado errado seria ignorado e a sessao gravaria uma configuracao diferente da
que o pesquisador acredita ter usado.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    """Fonte de video."""

    uri: str = "0"                     # arquivo, indice de webcam ou URL RTSP
    realtime: bool = False             # True = descarta quadros para manter latencia
    # Largura de PROCESSAMENTO. Tudo o que vai para o banco volta para os pixels
    # da imagem original, entao ground truth e calibracao nunca mudam de escala.
    resize_width: int | None = 1280    # None mantem a resolucao original
    target_fps: float | None = None    # limita a taxa de captura
    queue_size: int = 2
    start_frame: int = 0
    max_frames: int = 0                # 0 = sem limite
    # Instante real (ISO-8601, com fuso) do primeiro quadro de um video GRAVADO.
    # Sem ele, so ha tempo relativo ao inicio do video; com ele, cada evento ganha
    # media_time_utc. Em camera/stream ao vivo e preenchido automaticamente.
    recording_start: str | None = None
    # Decodificador de arquivos: "auto" = AVFoundation (hardware) no macOS, com
    # FFmpeg de reserva e nas buscas (start_frame); "ffmpeg" = software (o padrao
    # ate a 0.3). Um quadro 4K HEVC: 8 ms contra 46-48 ms.
    backend: str = "auto"
    # Prazo de latencia ponta a ponta, em ms, para contar quadros atrasados.
    # None = um periodo de quadro da fonte (1000/fps). Recalculavel na analise.
    deadline_ms: float | None = None
    # Limites nativos FFmpeg para abertura/leitura de streams de rede.
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if self.open_timeout_ms <= 0 or self.read_timeout_ms <= 0:
            raise ValueError("source open_timeout_ms e read_timeout_ms devem ser > 0")


@dataclass
class DetectorConfig:
    """Deteccao de veiculos e de placas."""

    vehicle_weights: str = "data/models/yolo11n.pt"
    plate_weights: str = "data/models/plate_yolo.pt"
    # Quais classes detectar: "veiculos" (so o que tem placa, mais rapido),
    # "trafego" (pessoas + veiculos) ou "completo" (inclui mobiliario urbano).
    classes: str = "trafego"
    plate_detector: str = "auto"       # "yolo" | "classic" | "auto"
    detect_vehicles: bool = True       # False = procura placas no quadro inteiro
    conf: float = 0.30
    plate_conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    plate_imgsz: int = 320
    device: str = "auto"               # "auto" | "coreml" | "mps" | "cuda" | "cpu"
    # Quem executa o YOLO: "auto" = CoreML direto para .mlpackage (pre e pos-
    # processamento proprios, ~2x mais rapido), ONNX Runtime para .onnx,
    # ultralytics para .pt. "ultralytics" forca o ultralytics em todos (referencia).
    runtime: str = "auto"
    half: bool = True
    # >1: detecta a cada N quadros; nos intermediarios o rastreador so preve
    # (sem medida, sem busca de placa, sem OCR). Reduz custo, nao latencia.
    detect_every_n: int = 1
    vehicle_pad: float = 0.02          # dilatacao do recorte do veiculo
    # Teto de veiculos em que se procura placa por quadro (os maiores primeiro).
    # 0 = sem teto. Sem teto, o custo de um quadro cresce com o numero de
    # veiculos e nao ha limite superior de latencia.
    max_rois_per_frame: int = 0
    # Fracao maxima da largura do veiculo que a placa ocupa. A placa so e
    # procurada em veiculos com largura x fracao >= ocr.min_plate_width: num
    # veiculo mais estreito a placa nao chega ao tamanho que o OCR le, e a busca
    # (~2,5 ms por veiculo com o YOLO de placas) seria descartada. Medido em
    # video real (p90 da placa/veiculo): carro 0,33, moto 0,38, caminhao 0,22,
    # onibus 0,14. Em video 4K de rua, 35% das buscas eram em veiculos distantes
    # assim. 0 = procura em todos (avaliacao da deteccao de placas).
    plate_width_ratio: float = 0.5
    # IoU acima do qual duas caixas de veiculo (carro/caminhao/onibus/moto) de
    # classes diferentes sao o mesmo veiculo: fica a de maior score. Sem isso,
    # o mesmo carro detectado como "car" e "truck" vira duas trilhas e a placa
    # e publicada duas vezes. 0 = desligado (comportamento da 0.3.0).
    vehicle_merge_iou: float = 0.7
    # Se o detector de objetos nao encontrar NADA que porte placa por muitos
    # quadros seguidos e houver placas visiveis, passa a rastrear as placas
    # diretamente. Desligue em experimentos que exigem comportamento fixo.
    auto_plate_fallback: bool = True
    fallback_after_frames: int = 60
    # Workers da etapa paralela (preparo do quadro, deteccao e, no perfil que le
    # toda placa, OCR antecipado): 0 = pipeline serial; -1 = automatico. Offline
    # o automatico usa os nucleos - o resultado e o do serial, bit a bit; em
    # tempo real fica serial: com a maquina acompanhando a camera, a reordenacao
    # so acrescentaria latencia. Em paralelo a latencia por quadro offline
    # inclui a fila e nao mede o custo de processamento (ver docs/PARALELISMO.md).
    workers: int = -1

    def __post_init__(self) -> None:
        if self.workers < -1:
            raise ValueError("detector.workers deve ser >= -1 (0 = serial, -1 = automatico)")


@dataclass
class TrackerConfig:
    """Limiares do rastreador.

    Os limiares de associacao sao **IoU minimo** (nao custo). Os padroes de
    veiculo reproduzem os valores efetivos do ByteTrack de referencia, cujo
    ``match_thresh=0.8`` e um custo maximo, isto e, IoU >= 0,2.

    Ha dois perfis porque os objetos rastreados tem escalas muito diferentes:

    * **veiculos** - caixas grandes; a primeira passada aceita IoU >= 0,2 e o
      algoritmo hungaro + Kalman resolvem a identidade.
    * **placas** (modo sem detector de veiculos) - caixas de ~60x20 px, cujos
      cantos oscilam alguns pixels por quadro, e scores heuristicos baixos no
      detector classico.

    ``max_age`` e contado em quadros da fonte.
    """

    high_thresh: float = 0.5
    low_thresh: float = 0.1
    match_iou: float = 0.2
    second_match_iou: float = 0.5
    max_age: int = 30
    min_hits: int = 3
    # Perfil aplicado quando as placas sao os objetos rastreados.
    plate_high_thresh: float = 0.30
    plate_low_thresh: float = 0.08
    plate_match_iou: float = 0.30
    plate_second_match_iou: float = 0.15


@dataclass
class OCRConfig:
    engine: str = "auto"               # auto|fastplate|paddle|easyocr|tesseract|template
    model_name: str = "cct-s-v1-global-model"
    # O modelo de OCR e pequeno e o CoreML o fatia em ~56 particoes, cujo custo
    # de troca de contexto supera o ganho. Roda em CPU por padrao; ative para
    # medir por conta propria (ver FastPlateOCR.__init__).
    force_coreml: bool = False
    min_confidence: float = 0.35
    # Largura minima da placa na imagem PROCESSADA (a da deteccao) para tentar o
    # OCR; abaixo disso a leitura e ruido. Medido em video 4K lido a 1280 px: de
    # 45 px em diante, 95% das leituras iguais a decisao final do veiculo; o
    # limiar antigo (60) descartava essa faixa. Com 35 px publica mais placas,
    # mas ja com falsos positivos (adesivos lidos como placa).
    min_plate_width: int = 45
    # O recorte para o OCR sai do quadro ORIGINAL quando a fonte foi reduzida
    # (resize_width): detecta na imagem pequena, le com os pixels da camera.
    # Medido em video 4K: placas de 35-45 px (a 1280) lidas como a decisao final
    # do veiculo em 83% das tentativas, contra 43% no recorte reduzido.
    full_resolution: bool = True
    ocr_every_n: int = 2               # roda OCR a cada N quadros por trilha
    max_ocr_per_frame: int = 4         # teto de custo por quadro (garantia de RT)
    # Depois de publicada, a trilha so continua sendo lida enquanto tiver menos
    # que N leituras. 0 = nunca para. ATENCAO: com N > 0 a amostra de leituras
    # depende do proprio acerto do motor (quem acerta cedo para de ser lido), o
    # que enviesa a exatidao por leitura e censura a refusao offline. O perfil de
    # avaliacao usa 0.
    max_reads_published: int = 12
    # Maximo de caracteres que a gramatica brasileira pode trocar (ver plates_br).
    max_grammar_corrections: int = 2
    workers: int = 2                   # threads de OCR (o GIL e liberado no ONNX/OpenCV)
    # Threads do onnxruntime POR sessao de OCR (fast-plate-ocr); 0 = padrao da
    # biblioteca (todos os nucleos). O numero muda o resultado no ultimo bit: com
    # 1 ou 2 threads as confiancas diferem em ~3e-7 das com 4 ou mais. Fixo, as
    # leituras nao dependem dos nucleos da maquina, e 4 reproduz o padrao do
    # Apple M5 Pro bit a bit. Sem espera ativa, cada leitura gasta 2,35x menos
    # CPU (27,7 contra 65 ms) por 1,3 ms a mais, CPU que a deteccao e a
    # decodificacao disputam.
    threads: int = 4
    pad: float = 0.06                  # margem no recorte da placa
    max_pending_jobs: int = 32         # teto global: inclui trabalhos ja obsoletos em voo
    job_timeout_s: float = 5.0         # resultados mais velhos nao entram na fusao

    def __post_init__(self) -> None:
        if self.max_pending_jobs < 1 or self.job_timeout_s <= 0:
            raise ValueError("ocr max_pending_jobs e job_timeout_s devem ser > 0")
        if self.threads < 0:
            raise ValueError("ocr.threads deve ser >= 0 (0 = padrao do onnxruntime)")


@dataclass
class FusionConfig:
    mode: str = "char"                 # "char" | "string"
    min_reads: int = 3
    min_agreement: float = 0.6
    min_confidence: float = 0.45
    require_valid_syntax: bool = True
    # Leituras mantidas por trilha (as de maior confianca). 0 = todas. O corte
    # muda a votacao; para avaliacao e refusao offline use 0.
    max_reads_per_track: int = 60
    # Quando publicar (ver lprts.fusion.publication): "first_stable" (primeira
    # fusao estavel, sem revisao), "revise" (revisa se uma fusao estavel
    # posterior mudar o texto) ou "on_exit" (decide quando a trilha termina).
    publish_policy: str = "first_stable"
    # > 0: trilha que termina sem decisao publica com ao menos N leituras (os
    # demais criterios valem). Marcado como "exit_relaxed". 0 = desligado.
    exit_min_reads: int = 0


@dataclass
class AnalyticsConfig:
    db_path: str = "runs/lprts.db"
    store_raw_reads: bool = True
    store_frame_telemetry: bool = True
    # Cada deteccao (objeto e placa) por quadro. Permite reavaliar a deteccao a
    # partir do banco, sem reprocessar o video.
    store_detections: bool = True
    # Estado de cada trilha em cada quadro (caixa, placa associada). E a
    # trajetoria completa e a base das metricas de rastreamento.
    store_track_points: bool = True
    save_crops: bool = True
    crops_dir: str = "runs/crops"
    # Cada processamento (Iniciar na interface, `lprts run`, `lprts eval`) comeca
    # do zero: apaga o banco db_path e os recortes das sessoes dele. As repeticoes
    # de um mesmo `lprts eval --repeat` ficam juntas. False acumula as execucoes.
    clear_previous: bool = True
    async_writes: bool = True
    batch_size: int = 256
    lines: list[dict] = field(default_factory=list)         # linhas virtuais
    calibration: dict = field(default_factory=dict)          # homografia/escala


@dataclass
class UIConfig:
    enabled: bool = True
    window_title: str = "LPR-RTS - Reconhecimento de Placas em Tempo Real"
    max_width: int = 1280
    show_vehicle_boxes: bool = True
    show_plate_boxes: bool = True
    show_tracks: bool = True
    show_trajectories: bool = False
    show_hud: bool = True
    theme: str = "dark"


@dataclass
class AppConfig:
    """Configuracao raiz."""

    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log_level: str = "INFO"

    # -- serializacao ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash estavel da configuracao que afeta resultados (agrupa execucoes iguais).

        Exclui o que nao muda o resultado: caminho do banco, interface, log.
        """
        dados = self.to_dict()
        dados.pop("ui", None)
        dados.pop("log_level", None)
        dados.get("analytics", {}).pop("db_path", None)
        dados.get("analytics", {}).pop("crops_dir", None)
        dados.get("analytics", {}).pop("clear_previous", None)
        dados.get("source", {}).pop("uri", None)
        texto = json.dumps(dados, sort_keys=True, default=str)
        return hashlib.sha256(texto.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return _build(cls, data or {}, "")

    @classmethod
    def load(cls, path: str | Path | None) -> AppConfig:
        """Carrega de YAML; sem caminho, devolve os padroes."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {path}")
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    @classmethod
    def load_many(cls, paths: Sequence[str | Path]) -> AppConfig:
        """Carrega varios YAMLs em sequencia; os posteriores sobrepoem os anteriores.

        Permite combinar um perfil (``configs/avaliacao.yaml``) com a calibracao
        de uma camera (``configs/minha_camera.yaml``) sem editar nenhum dos dois.
        """
        merged: dict[str, Any] = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {p}")
            _deep_merge(merged, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        return cls.from_dict(merged)

    def override(self, **kwargs: Any) -> AppConfig:
        """Aplica sobreposicoes com caminho pontilhado: ``override(**{"ocr.engine": "paddle"})``.

        Valores ``None`` sao ignorados, o que permite passar direto os argumentos
        opcionais da CLI sem precisar filtrar quais foram informados.
        """
        for dotted, value in kwargs.items():
            if value is None:
                continue
            target: Any = self
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            if not hasattr(target, parts[-1]):
                raise AttributeError(f"Opcao de configuracao desconhecida: {dotted}")
            setattr(target, parts[-1], value)
        return self


def _deep_merge(base: dict, extra: dict) -> dict:
    for chave, valor in extra.items():
        if isinstance(valor, dict) and isinstance(base.get(chave), dict):
            _deep_merge(base[chave], valor)
        else:
            base[chave] = valor
    return base


# Com `from __future__ import annotations`, os campos das dataclasses guardam o
# tipo como *string*, entao `is_dataclass(f.type)` nunca acerta. Resolvemos os
# tipos reais via get_type_hints para reconstruir a config aninhada.
def _resolve_nested(cls: type) -> dict[str, type]:
    import typing

    hints = typing.get_type_hints(cls)
    return {name: hint for name, hint in hints.items() if isinstance(hint, type) and is_dataclass(hint)}


_NESTED: dict[type, dict[str, type]] = {
    AppConfig: _resolve_nested(AppConfig),
}

# Chaves renomeadas: em vez de reinterpretar em silencio (a semantica mudou),
# falhamos com a explicacao.
_RENOMEADAS = {
    "tracker.match_thresh": (
        "tracker.match_iou",
        "a semantica mudou: agora e IoU MINIMO; o antigo 0.8 equivale a match_iou: 0.8, "
        "mas o padrao recomendado (equivalente ao ByteTrack) e 0.2",
    ),
    "tracker.second_match_thresh": ("tracker.second_match_iou", "IoU minimo da 2a passada"),
    "tracker.plate_match_thresh": ("tracker.plate_match_iou", "IoU minimo (perfil de placas)"),
    "tracker.plate_second_match_thresh": (
        "tracker.plate_second_match_iou", "IoU minimo da 2a passada (perfil de placas)",
    ),
}


class ConfigError(ValueError):
    """Configuracao invalida (chave desconhecida ou renomeada)."""


def _build(cls: type, data: dict[str, Any], prefix: str) -> Any:
    """Reconstroi dataclasses aninhadas a partir de dicionarios YAML."""
    if not isinstance(data, dict):
        raise ConfigError(f"Secao '{prefix or 'raiz'}' deveria ser um mapeamento, veio {data!r}")
    nested = _NESTED.get(cls) or _resolve_nested(cls)
    validos = {f.name for f in fields(cls)}
    for chave in data:
        if chave in validos:
            continue
        caminho = f"{prefix}.{chave}" if prefix else chave
        if caminho in _RENOMEADAS:
            novo, motivo = _RENOMEADAS[caminho]
            raise ConfigError(f"Opcao '{caminho}' foi renomeada para '{novo}' ({motivo}).")
        parecidas = difflib.get_close_matches(chave, sorted(validos), n=3)
        dica = f" Voce quis dizer: {', '.join(parecidas)}?" if parecidas else ""
        raise ConfigError(f"Opcao de configuracao desconhecida: '{caminho}'.{dica}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        sub = nested.get(f.name)
        caminho = f"{prefix}.{f.name}" if prefix else f.name
        if sub is not None:
            kwargs[f.name] = _build(sub, value or {}, caminho)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
