"""Orquestrador do pipeline de LPR.

Fluxo por quadro::

    captura -> deteccao de objetos -> deteccao de placas (nas ROIs) -> rastreamento
            -> agendamento de OCR (assincrono) -> fusao multi-frame -> analytics -> banco

O processamento de um quadro tem duas etapas com naturezas opostas:

* ``detect_stage`` - **sem estado**: deteccao de objetos e de placas. Pode rodar
  em paralelo em varios quadros (``core.parallel``).
* ``sequential_stage`` - **com estado e em ordem**: rastreamento, OCR, fusao,
  analitica e gravacao. Serial e paralelo usam exatamente este codigo, o que
  garante resultados identicos.

Tres decisoes que fazem o sistema operar em *tempo real brando* (soft real-time)
- e nao apenas rapido:

1. **OCR assincrono.** O OCR roda num pool de threads e os resultados sao
   colhidos em quadros seguintes; o laco principal nunca espera. Cada leitura e
   gravada com o quadro de onde o recorte saiu, nao com o quadro em que o
   resultado chegou.
2. **Orcamento por quadro.** ``max_ocr_per_frame`` limita as leituras e
   ``max_rois_per_frame`` limita a busca de placas por quadro; veiculos
   estreitos demais para uma placa legivel nao tem a placa procurada
   (``plate_width_ratio``).
3. **Descarte a montante.** Quadros sao descartados na captura
   (``DropOldestQueue``), nunca no meio do processamento. Os descartes e a
   latencia ponta a ponta (captura -> resultado) sao gravados por quadro.

Nada disso garante prazo: o sistema nao e de tempo real rigido. O que ele faz
e degradar de forma controlada e **medir** o quanto degradou.

No modo offline (``realtime=False``) nada e descartado e o OCR e sincrono, o que
torna a avaliacao deterministica e reproduzivel.

Ciclo de vida das trilhas e dos dados
-------------------------------------
* Uma trilha e gravada em ``tracks`` quando TERMINA (sai de cena, fim do video
  ou troca de perfil), com o estado completo: permanencia, trajetoria, todas as
  leituras. Trilhas sem placa (pedestres, ciclistas) tambem - e o que permite
  medir a composicao do trafego.
* Um *evento* e gravado quando a fusao atinge o criterio de publicacao - sempre
  o mesmo criterio, inclusive no fim do video.
* Tempos fisicos usam o tempo de midia (``Frame.source_pts``); coordenadas
  gravadas estao em pixels da imagem original.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analytics.metrics import MetricsCollector
from ..analytics.store import ResultStore
from ..analytics.traffic import Calibration, TrafficAnalyzer, VirtualLine
from ..capture.source import ImageFolderSource, VideoSource
from ..config import AppConfig
from ..detect.backends import describe_environment, resolve_device
from ..detect.base import Detector, merge_vehicle_duplicates
from ..fusion.publication import Decision, build_policy
from ..fusion.voting import FusedPlate, PlateVoter
from ..ocr.base import OCREngine
from ..track.bytetrack import ByteTracker
from ..utils.geometry import clip_bbox, containment_ratio, expand_bbox
from ..utils.imgproc import brightness_contrast, crop, crop_rotated, sharpness
from ..utils.logging import get_logger
from .clock import Stopwatch
from .taxonomy import has_plate
from .types import (
    Detection,
    Frame,
    FrameRef,
    FrameResult,
    PlateRead,
    StageTiming,
    TrackState,
)

log = get_logger(__name__)


class LPRPipeline:
    """Pipeline completo de reconhecimento de placas."""

    def __init__(
        self,
        config: AppConfig,
        vehicle_detector: Detector | None = None,
        plate_detector: Detector | None = None,
        ocr_engine: OCREngine | None = None,
        store: ResultStore | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.detector.device)

        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self.store = store
        if ocr_engine is not None:
            ocr_engine.max_corrections = config.ocr.max_grammar_corrections

        # O perfil do rastreador depende do que esta sendo rastreado: caixas de
        # placa sao pequenas e tremidas e exigem limiares proprios.
        plate_mode = vehicle_detector is None or not config.detector.detect_vehicles
        self.tracker = ByteTracker(**self._tracker_profile(plate_mode))
        self.voter = PlateVoter(
            min_reads=config.fusion.min_reads,
            min_agreement=config.fusion.min_agreement,
            min_confidence=config.fusion.min_confidence,
            require_valid_syntax=config.fusion.require_valid_syntax,
            mode=config.fusion.mode,
            max_reads_per_track=config.fusion.max_reads_per_track,
            max_corrections=config.ocr.max_grammar_corrections,
        )
        self.policy = build_policy(self.voter, config.fusion)
        cal = config.analytics.calibration or {}
        tamanho = cal.get("image_size")
        self.traffic = TrafficAnalyzer(
            lines=[
                VirtualLine(linha["name"], tuple(linha["p1"]), tuple(linha["p2"]))
                for linha in config.analytics.lines
            ],
            calibration=Calibration(
                image_points=[tuple(p) for p in cal.get("image_points", [])],
                world_points=[tuple(p) for p in cal.get("world_points", [])],
                meters_per_pixel=cal.get("meters_per_pixel"),
                image_size=tuple(tamanho) if tamanho else None,
            ),
        )
        self.metrics = MetricsCollector()

        # Estado do agendamento de OCR
        self._ocr_pool: ThreadPoolExecutor | None = None
        self._pending: dict[int, Future] = {}          # track_id -> leitura em voo
        self._pending_since: dict[int, float] = {}
        self._expired_pending: set[int] = set()
        self._last_ocr_frame: dict[int, int] = {}      # track_id -> ultimo quadro lido
        self._published: set[int] = set()              # trilhas que ja geraram evento
        # Trilhas encerradas pelo rastreador mas com OCR ainda em voo: so sao
        # gravadas quando a leitura chega, para que ela entre na fusao final.
        self._ending: dict[int, TrackState] = {}
        # Estado do fallback automatico para o modo so-placa (ver _check_blind).
        self._blind_frames = 0
        self._fallback_active = False
        self._fallback_reason = ""
        self._stop = threading.Event()
        self._source: Any = None
        self._closed_components: set[int] = set()
        self._component_close_lock = threading.Lock()
        self._frame_index = 0
        self._sched_lock = threading.Lock()
        self._last_detection_index: int | None = None
        self._prev_index: int | None = None
        self._t0: float | None = None                 # relogio do 1o quadro (tempos relativos)
        self._last_ref: FrameRef | None = None
        self._deadline_ms: float | None = config.source.deadline_ms
        self._stats: dict[str, int] = {
            "tracks_closed": 0, "tracks_discarded_tentative": 0,
            "ocr_failures": 0, "ocr_lost_at_end": 0, "orphan_reads": 0,
            "ocr_backpressure": 0, "ocr_expired": 0,
            # OCR antecipado pelos workers do pipeline paralelo (ver core.parallel):
            # quantas leituras a etapa serial aproveitou.
            "ocr_prefetch_used": 0,
        }

        self._crops_dir = Path(config.analytics.crops_dir)
        if config.analytics.save_crops:
            self._crops_dir.mkdir(parents=True, exist_ok=True)

    def _tracker_profile(self, plate_mode: bool) -> dict[str, Any]:
        tk = self.config.tracker
        if plate_mode:
            return {
                "high_thresh": tk.plate_high_thresh, "low_thresh": tk.plate_low_thresh,
                "match_iou": tk.plate_match_iou, "second_match_iou": tk.plate_second_match_iou,
                "max_age": tk.max_age, "min_hits": tk.min_hits,
            }
        return {
            "high_thresh": tk.high_thresh, "low_thresh": tk.low_thresh,
            "match_iou": tk.match_iou, "second_match_iou": tk.second_match_iou,
            "max_age": tk.max_age, "min_hits": tk.min_hits,
        }

    # -- ciclo de vida -----------------------------------------------------

    def start(self) -> LPRPipeline:
        if self.config.ocr.max_pending_jobs < 1 or self.config.ocr.job_timeout_s <= 0:
            raise ValueError("Limites de trabalhos OCR devem ser > 0")
        workers = max(1, self.config.ocr.workers)
        if self.config.source.realtime and self.ocr_engine is not None and self._ocr_pool is None:
            self._ocr_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr")
        return self

    def request_stop(self) -> None:
        """Cancelamento nao bloqueante; pode ser chamado pela interface."""
        self._stop.set()
        cancel = getattr(self._source, "cancel", None)
        if callable(cancel):
            cancel()

    def _close_component(self, component: Any) -> None:
        if component is None:
            return
        with self._component_close_lock:
            if id(component) in self._closed_components:
                return
            self._closed_components.add(id(component))
        try:
            component.close()
        except Exception as exc:  # noqa: BLE001 - liberar os demais componentes mesmo assim
            log.debug("Falha ao liberar %s: %s", type(component).__name__, exc)

    def stop(self) -> None:
        """Encerra o pipeline e libera os modelos de forma ordenada.

        A ordem importa: o pool de OCR e encerrado ANTES dos motores, senao uma
        tarefa em voo usaria uma sessao ja liberada. Sessoes de inferencia
        seguram recursos nativos (CoreML, Metal, CUDA); deixar que sejam
        coletadas em ordem arbitraria pelo garbage collector, ja durante o
        encerramento do interpretador, e uma fonte conhecida de falha de
        segmentacao no macOS.
        """
        self.request_stop()
        if self._ocr_pool is not None:
            self._ocr_pool.shutdown(wait=False, cancel_futures=True)
            running = [future for future in self._pending.values() if future.running()]
            if running:
                _, unfinished = wait(running, timeout=self.config.ocr.job_timeout_s)
                if unfinished:
                    # Nao liberar uma sessao nativa ainda usada por uma thread.
                    # Liberar apos a ULTIMA leitura sair, mesmo se o chamador
                    # nao repetir stop. Uma tarefa nativa travada para sempre
                    # exige supervisao/reinicio do processo.
                    def close_when_idle(_future: Future) -> None:
                        if all(future.done() for future in unfinished):
                            self._close_component(self.ocr_engine)

                    for future in unfinished:
                        future.add_done_callback(close_when_idle)
                    for detector in (self.plate_detector, self.vehicle_detector):
                        self._close_component(detector)
                    raise TimeoutError("OCR ainda em execucao apos cancelamento")
            self._ocr_pool = None
        self._pending.clear()
        self._pending_since.clear()
        self._expired_pending.clear()

        for componente in (self.ocr_engine, self.plate_detector, self.vehicle_detector):
            self._close_component(componente)

    def __enter__(self) -> LPRPipeline:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        try:
            self.finalize()
        finally:
            self.stop()

    # -- deteccao ----------------------------------------------------------

    @property
    def _plate_only_mode(self) -> bool:
        """True quando as **placas** sao os objetos rastreados.

        Vale quando nao ha detector de objetos habilitado, ou quando o fallback
        automatico foi acionado (ver ``_check_blind``).

        Rastrear um retangulo de quadro inteiro faria todas as placas da cena
        cairem numa unica trilha e votarem juntas, produzindo uma placa quimera -
        por isso o modo existe explicitamente.
        """
        if self._fallback_active:
            return True
        return self.vehicle_detector is None or not self.config.detector.detect_vehicles

    def should_detect(self, frame_index: int) -> bool:
        """Decide se este quadro roda deteccao (``detect_every_n``).

        A decisao usa a distancia ate a ultima deteccao, nao ``indice % n``: com
        quadros descartados, os indices processados poderiam cair sempre fora do
        multiplo e a deteccao nunca rodaria. Chamado em ordem de quadro (o
        pipeline paralelo chama sob a trava de retirada da fila).
        """
        n = max(1, self.config.detector.detect_every_n)
        with self._sched_lock:
            if (
                n <= 1
                or self._last_detection_index is None
                or frame_index - self._last_detection_index >= n
                or frame_index < self._last_detection_index
            ):
                self._last_detection_index = frame_index
                return True
            return False

    def detect_stage(
        self,
        image: np.ndarray,
        run_detection: bool = True,
        detectors: tuple[Detector | None, Detector | None] | None = None,
    ) -> tuple[list[Detection], list[Detection], int]:
        """Etapa SEM estado: objetos e placas de um quadro.

        Devolve ``(objetos, placas, rois_puladas)``. Os detectores podem vir por
        parametro (instancias proprias de cada worker paralelo).
        """
        if not run_detection:
            return [], [], 0
        vdet, pdet = detectors if detectors is not None else (
            self.vehicle_detector, self.plate_detector
        )
        if self._plate_only_mode:
            return [], self._detect_plates(image, [], detector=pdet), 0
        objetos = vdet.detect(image) if vdet is not None else []
        # Mesmo veiculo como "car" e "truck": uma trilha so (ver merge_vehicle_duplicates).
        objetos = merge_vehicle_duplicates(objetos, self.config.detector.vehicle_merge_iou)
        rois, puladas = self._select_rois(objetos)
        return objetos, self._detect_plates(image, rois, detector=pdet), puladas

    def _select_rois(self, objects: list[Detection]) -> tuple[list[Detection], int]:
        """Objetos que podem portar placa e legivel, os maiores primeiro, ate o teto.

        Devolve ``(rois, puladas)``; ``puladas`` soma os veiculos estreitos demais
        para uma placa legivel (``detector.plate_width_ratio``) e os acima do teto
        (``max_rois_per_frame``). As larguras ficam em ``detections``: os dois
        motivos se separam na analise.
        """
        candidatos = [o for o in objects if has_plate(o.label)]
        pulados = 0
        razao = self.config.detector.plate_width_ratio
        if razao > 0:
            minimo = self.config.ocr.min_plate_width / razao
            legiveis = [o for o in candidatos if o.bbox[2] - o.bbox[0] >= minimo]
            pulados, candidatos = len(candidatos) - len(legiveis), legiveis
        teto = self.config.detector.max_rois_per_frame
        if teto > 0 and len(candidatos) > teto:
            candidatos.sort(key=lambda d: d.area, reverse=True)
            return candidatos[:teto], pulados + len(candidatos) - teto
        return candidatos, pulados

    def _detect_plates(
        self,
        image: np.ndarray,
        vehicles: list[Detection],
        detector: Detector | None = None,
    ) -> list[Detection]:
        """Procura placas dentro de cada ROI de veiculo (ou no quadro inteiro).

        Buscar dentro da ROI e mais rapido *e* mais preciso: a placa ocupa uma
        fracao maior da entrada do detector, e falsos positivos fora de veiculos
        (letreiros, placas de rua) desaparecem por construcao.
        """
        detector = detector or self.plate_detector
        if detector is None:
            return []
        h, w = image.shape[:2]
        plates: list[Detection] = []

        if self._plate_only_mode:
            return detector.detect(image)
        if not vehicles:
            # Sem veiculos nao ha a que associar uma placa. O caso de o detector
            # de objetos estar cego e tratado em _check_blind.
            return []

        for vehicle in vehicles:
            # Pedestres, bicicletas e mobiliario urbano nao portam placa.
            if not has_plate(vehicle.label):
                continue
            box = expand_bbox(vehicle.bbox, self.config.detector.vehicle_pad, w, h)
            roi = crop(image, box)
            if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
                continue
            offset_x, offset_y = box[0], box[1]
            for det in detector.detect(roi):
                x1, y1, x2, y2 = det.bbox
                plates.append(
                    Detection(
                        bbox=clip_bbox(
                            (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), w, h
                        ),
                        score=det.score, label="plate", class_id=0,
                    )
                )
        return plates

    def _check_blind(
        self, image: np.ndarray, vehicles: list[Detection]
    ) -> list[Detection] | None:
        """Detecta o detector de objetos inutil para este video e cai para so-placa.

        Motivacao concreta: um detector COCO nao reconhece veiculos estilizados
        (video sintetico, animacao, imagem termica). Sem isto o sistema entra
        num modo de falha *silencioso*: placas ate seriam detectadas, mas nao
        existe trilha a que associa-las.

        O criterio conta quadros sem nenhum objeto **que possa portar placa**. A
        troca so acontece se houver placas de fato no quadro. Devolve as placas
        da sondagem quando o fallback e acionado (para usa-las neste quadro), ou
        ``None``.
        """
        cfg = self.config.detector
        if self._fallback_active or not cfg.auto_plate_fallback:
            return None
        if self.plate_detector is None:
            return None

        if any(has_plate(v.label) for v in vehicles):
            self._blind_frames = 0
            return None

        self._blind_frames += 1
        if self._blind_frames < max(10, cfg.fallback_after_frames):
            return None

        # Sondagem: ha placas visiveis apesar de nenhum objeto detectado?
        placas = self.plate_detector.detect(image)
        self._blind_frames = 0
        if not placas:
            return None

        nome = self.vehicle_detector.name if self.vehicle_detector else "?"
        vistos = sorted({v.label for v in vehicles})
        contexto = (
            f"so reconheceu {', '.join(vistos)}"
            if vistos
            else "nao reconheceu nenhum veiculo"
        )
        self._fallback_active = True
        self._fallback_reason = (
            f"O detector de objetos ({nome}) {contexto} em "
            f"{cfg.fallback_after_frames} quadros seguidos. Pode haver veiculos na "
            "cena: este detector foi treinado em fotos reais e nao reconhece formas "
            f"sinteticas ou estilizadas. Como {len(placas)} placa(s) foram vistas, "
            "passando a rastrear as placas diretamente."
        )
        # Placas tem outra escala e outros limiares. As trilhas de veiculo em
        # curso sao encerradas (motivo "reset") e gravadas normalmente.
        self.tracker.reset()
        self._apply_plate_tracker_profile()
        if self.store is not None:
            self.store.add_note(
                f"FALLBACK AUTOMATICO no quadro {self._frame_index}: {self._fallback_reason}"
            )
        log.warning("%s", self._fallback_reason)
        log.warning(
            "Para fixar esse comportamento e evitar a deteccao inutil de objetos, "
            "rode com: --no-vehicle-detector"
        )
        return placas

    def _apply_plate_tracker_profile(self) -> None:
        """Troca os limiares do rastreador para o perfil de placas (caixas pequenas)."""
        perfil = self._tracker_profile(plate_mode=True)
        for chave, valor in perfil.items():
            setattr(self.tracker, chave, valor)

    @property
    def fallback_active(self) -> bool:
        """True se o pipeline trocou automaticamente para o modo so-placa."""
        return self._fallback_active

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    def _assign_plates(
        self, plates: list[Detection], tracks: list[TrackState]
    ) -> dict[int, Detection]:
        """Associa cada placa a trilha correspondente.

        No modo so-placa a correspondencia e direta (a trilha *e* a placa); no
        modo com veiculos, cada placa vai para a trilha de veiculo que a contem.
        """
        if self._plate_only_mode:
            # Usa a caixa da trilha, ja suavizada pelo Kalman: menos tremida que
            # a deteccao crua, o que estabiliza o recorte enviado ao OCR.
            return {
                t.track_id: Detection(t.bbox, t.score, "placa_veicular", 0, t.track_id)
                for t in tracks
            }

        # So veiculos que portam placa a recebem. Antes a caixa do piloto de uma
        # moto (ou de um ciclista, de um pedestre na frente do carro) tambem
        # contava: a placa ia para a trilha da pessoa, que nunca passa pelo OCR,
        # e a moto ficava sem leitura.
        portadores = [t for t in tracks if t.can_have_plate]
        assignment: dict[int, Detection] = {}
        for plate in plates:
            razoes = [(containment_ratio(plate.bbox, t.bbox), t) for t in portadores]
            best_ratio = max((r for r, _ in razoes), default=0.0)
            if best_ratio < 0.5:
                continue
            # Placa contida (quase) por inteiro em mais de um veiculo - caixas de
            # carros vizinhos se sobrepoem em perspectiva: fica com o veiculo cujo
            # CENTRO horizontal esta mais perto dela (a placa traseira e centrada).
            # Antes o empate era decidido pela ordem da lista, que muda de quadro a
            # quadro: as duas trilhas liam a mesma placa e a publicavam duas vezes -
            # ou um carro era publicado com a placa do vizinho.
            cx = (plate.bbox[0] + plate.bbox[2]) / 2
            best_track = min(
                (t for r, t in razoes if r >= best_ratio - 0.05),
                key=lambda t: abs(cx - (t.bbox[0] + t.bbox[2]) / 2) / max(t.bbox[2] - t.bbox[0], 1.0),
            )
            current = assignment.get(best_track.track_id)
            if current is None or plate.score > current.score:
                if current is not None:
                    current.track_id = None
                plate.track_id = best_track.track_id
                assignment[best_track.track_id] = plate
        return assignment

    # -- OCR ---------------------------------------------------------------

    def _should_read(self, track: TrackState, plate: Detection) -> bool:
        """Decide se vale gastar OCR nesta trilha neste quadro."""
        cfg = self.config.ocr
        # No modo com detector de objetos, a trilha carrega a classe: uma pessoa
        # ou bicicleta nunca chega ao OCR, liberando orcamento para quem tem placa.
        if not self._plate_only_mode and not track.can_have_plate:
            return False
        # Largura em pixels da imagem PROCESSADA: e o que o OCR efetivamente ve.
        width = plate.bbox[2] - plate.bbox[0]
        if width < cfg.min_plate_width:
            return False   # abaixo disso o OCR devolve ruido
        if track.track_id in self._pending:
            return False   # ja existe uma leitura em voo
        last = self._last_ocr_frame.get(track.track_id, -10**9)
        if self._frame_index - last < cfg.ocr_every_n:
            return False
        # Trilha ja publicada e com leituras de sobra: parar de gastar OCR nela.
        # (Censura a amostra de leituras - o perfil de avaliacao desliga.)
        if (
            cfg.max_reads_published > 0
            and track.track_id in self._published
            and track.n_reads_total >= cfg.max_reads_published
        ):
            return False
        return True

    def _select_for_ocr(
        self, tracks: list[TrackState], assignment: dict[int, Detection]
    ) -> list[tuple[TrackState, Detection]]:
        """Escolhe as trilhas a ler neste quadro, respeitando o orcamento."""
        candidates: list[tuple[float, TrackState, Detection]] = []
        for track in tracks:
            plate = assignment.get(track.track_id)
            if plate is None or not self._should_read(track, plate):
                continue
            # Prioridade: placa maior (mais pixels = mais informacao) e trilha
            # ainda sem leitura estavel.
            area = (plate.bbox[2] - plate.bbox[0]) * (plate.bbox[3] - plate.bbox[1])
            unresolved = 0.0 if track.track_id in self._published else 1.0
            candidates.append((area * (1.0 + unresolved), track, plate))
        candidates.sort(key=lambda c: c[0], reverse=True)
        budget = max(1, self.config.ocr.max_ocr_per_frame)
        return [(t, p) for _, t, p in candidates[:budget]]

    def _plate_patch(self, frame: Frame, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Recorte da placa para o OCR (copia propria), na maior resolucao disponivel.

        A deteccao roda na imagem reduzida; com ``ocr.full_resolution`` o recorte
        sai do quadro original. Medido em video 4K (placa a 1280 px de largura,
        concordancia da leitura com a decisao final do veiculo): 35-45 px, 43% no
        recorte reduzido e 83% no original; 25-35 px, 0% e 42%; a partir de 45 px,
        iguais (95%).
        """
        pad = self.config.ocr.pad
        if frame.full_image is not None and 0 < frame.scale < 1.0:
            s = frame.scale
            caixa = (bbox[0] / s, bbox[1] / s, bbox[2] / s, bbox[3] / s)
            if frame.full_rotation is not None:
                return crop_rotated(frame.full_image, caixa, frame.full_rotation, pad=pad)
            return np.array(crop(frame.full_image, caixa, pad=pad), copy=True)
        return np.array(crop(frame.image, bbox, pad=pad), copy=True)

    @property
    def reads_every_plate(self) -> bool:
        """O agendamento le toda placa legivel em todo quadro (perfil de avaliacao).

        So entao vale ler antes de saber a trilha: fora disso, ``ocr_every_n`` e
        ``max_reads_published`` descartariam a maior parte (7x no perfil padrao).
        """
        cfg = self.config.ocr
        return cfg.ocr_every_n <= 1 and cfg.max_reads_published == 0

    def prefetch_ocr(
        self, frame: Frame, plates: list[Detection], engine: OCREngine,
    ) -> dict[int, PlateRead]:
        """Le, sem estado, cada placa que o agendamento poderia escolher neste quadro.

        A leitura so depende do recorte (quadro e caixa da deteccao), entao pode
        rodar num worker antes da etapa serial; ``_submit_ocr`` usa a leitura
        pronta da MESMA deteccao (por identidade) e o resultado e o do serial.
        No modo so-placa o recorte sai da caixa da trilha, que so a etapa serial
        conhece: nada e antecipado.
        """
        if self._plate_only_mode:
            return {}
        minimo = self.config.ocr.min_plate_width
        lidas: dict[int, PlateRead] = {}
        for plate in plates:
            if plate.bbox[2] - plate.bbox[0] < minimo:
                continue
            patch = self._plate_patch(frame, plate.bbox)
            lidas[id(plate)] = self._run_ocr(
                patch, plate.bbox, frame.index, frame.source_pts, time.monotonic(), engine=engine
            )
        return lidas

    def _submit_ocr(
        self, track: TrackState, plate: Detection, frame: Frame,
        prefetched: dict[int, PlateRead] | None = None,
    ) -> PlateRead | None:
        """Agenda (tempo real) ou executa (offline) o OCR de uma placa."""
        assert self.ocr_engine is not None
        if self._ocr_pool is not None and len(self._pending) >= self.config.ocr.max_pending_jobs:
            self._stats["ocr_backpressure"] += 1
            return None
        self._last_ocr_frame[track.track_id] = frame.index
        if self._ocr_pool is None and prefetched:
            pronta = prefetched.get(id(plate))
            if pronta is not None:
                self._stats["ocr_prefetch_used"] += 1
                return pronta
        # So o recorte atravessa para a thread de OCR - copia pequena e propria,
        # em vez da imagem inteira do quadro.
        patch = self._plate_patch(frame, plate.bbox)
        submetido = time.monotonic()
        if self._ocr_pool is not None:
            self._pending_since[track.track_id] = submetido
            self._pending[track.track_id] = self._ocr_pool.submit(
                self._run_ocr, patch, plate.bbox, frame.index, frame.source_pts, submetido
            )
            return None
        return self._run_ocr(patch, plate.bbox, frame.index, frame.source_pts, submetido)

    def _run_ocr(
        self,
        patch: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_index: int,
        video_time_s: float,
        submitted_at: float,
        engine: OCREngine | None = None,
    ) -> PlateRead:
        """Executa o OCR num recorte (no pool em tempo real; ``engine`` = o do worker)."""
        motor = engine or self.ocr_engine
        assert motor is not None
        inicio = time.monotonic()
        if (engine is None and self._ocr_pool is not None
                and inicio - submitted_at > self.config.ocr.job_timeout_s):
            raise TimeoutError("Trabalho OCR expirou antes de iniciar")
        if patch.size == 0:
            read = PlateRead(text="", confidence=0.0, engine=motor.name)
        else:
            read = motor.read(patch)
            read.sharpness = sharpness(patch)
            read.brightness, read.contrast = brightness_contrast(patch)
        read.bbox = bbox
        read.frame_index = frame_index
        read.video_time_s = video_time_s
        read.wait_ms = (inicio - submitted_at) * 1000.0
        return read

    def _incorporate(self, track: TrackState, read: PlateRead, ref: FrameRef) -> None:
        """Entrega uma tentativa de OCR a fusao e ao banco."""
        read.collected_frame = ref.index
        self.metrics.on_attempt()
        if read.text:
            self.voter.add_read(track, read)
            self.metrics.on_read(read.valid_syntax)
        if self.store is not None:
            self.store.add_read(track.track_id, read, scale=ref.scale)

    def _collect_ocr(self, ref: FrameRef) -> list[PlateRead]:
        """Colhe as leituras assincronas prontas desde o quadro anterior."""
        done: list[PlateRead] = []
        now = time.monotonic()
        for track_id, future in list(self._pending.items()):
            submitted = self._pending_since.get(track_id)
            if (submitted is not None and now - submitted > self.config.ocr.job_timeout_s
                    and track_id not in self._expired_pending):
                self._expired_pending.add(track_id)
                self._stats["ocr_expired"] += 1
            if not future.done():
                # Mesmo vencido continua ocupando capacidade: cancelar Future
                # nao remove seu recorte da fila interna de ThreadPoolExecutor.
                continue
            del self._pending[track_id]
            self._pending_since.pop(track_id, None)
            if track_id in self._expired_pending:
                self._expired_pending.remove(track_id)
                continue
            try:
                read = future.result()
            except Exception as exc:  # noqa: BLE001 - uma falha de OCR nao derruba o pipeline
                self._stats["ocr_failures"] += 1
                log.debug("OCR falhou na trilha %d: %s", track_id, exc)
                continue
            # A trilha pode estar temporariamente perdida (viva no rastreador,
            # fora da lista do quadro) ou encerrada esperando esta leitura.
            track = self.tracker.get(track_id) or self._ending.get(track_id)
            if track is None:
                self._stats["orphan_reads"] += 1
                read.collected_frame = ref.index
                if self.store is not None:
                    self.store.add_read(track_id, read, scale=ref.scale)
                continue
            self._incorporate(track, read, ref)
            if read.text:
                done.append(read)
        return done

    # -- laco principal ----------------------------------------------------

    def process_frame(self, frame: Frame, dropped: int = 0) -> FrameResult:
        """Processa um quadro de ponta a ponta.

        ``dropped`` e o total de quadros descartados pela fonte ate aqui; vai
        para o banco junto com a telemetria deste quadro.
        """
        t_start = time.monotonic()
        run = self.should_detect(frame.index)
        with Stopwatch() as sw:
            objetos, placas, puladas = self.detect_stage(frame.image, run)
        return self.sequential_stage(frame, objetos, placas, sw.ms, run, puladas, dropped, t_start)

    def sequential_stage(
        self,
        frame: Frame,
        vehicles: list[Detection],
        plates: list[Detection],
        detect_ms: float,
        run_detection: bool,
        rois_skipped: int = 0,
        dropped: int = 0,
        t_start: float | None = None,
        prefetched: dict[int, PlateRead] | None = None,
    ) -> FrameResult:
        """Etapa COM estado: rastreamento, OCR, fusao, analitica e gravacao.

        ``prefetched``: leituras ja feitas pelo worker paralelo, por deteccao.
        """
        t_start = t_start if t_start is not None else time.monotonic()
        if self._t0 is None:
            self._t0 = frame.timestamp
        self._frame_index = frame.index
        skipped = 0 if self._prev_index is None else max(0, frame.index - self._prev_index - 1)
        self._prev_index = frame.index
        ref = FrameRef(frame.index, frame.source_pts, frame.scale)
        timing = StageTiming(
            capture_ms=frame.decode_ms,
            queue_ms=max(0.0, (t_start - frame.timestamp) * 1000.0),
            detect_ms=detect_ms,
        )

        if run_detection and not self._plate_only_mode:
            with Stopwatch() as sw_blind:
                sondagem = self._check_blind(frame.image, vehicles)
            timing.detect_ms += sw_blind.ms
            if sondagem is not None:
                vehicles, plates = [], sondagem

        with Stopwatch() as sw_track:
            if run_detection:
                rastreados = plates if self._plate_only_mode else vehicles
                tracks = self.tracker.update(rastreados, frame.index, frame.source_pts)
            else:
                # Quadro intermediario de detect_every_n: so a predicao.
                tracks = self.tracker.predict_only(frame.index, frame.source_pts)
            encerradas = self.tracker.pop_removed()
            for active in self.tracker.tracks:
                if active.first_available_monotonic is None:
                    active.first_available_monotonic = frame.timestamp
            # Disponibilizar as removidas antes de colher futuras prontas.
            self._end_tracks(encerradas, ref)
        timing.track_ms = sw_track.ms

        assignment = self._assign_plates(plates, tracks) if run_detection else {}

        reads: list[PlateRead] = []
        with Stopwatch() as sw_ocr:
            reads.extend(self._collect_ocr(ref))
            if run_detection and self.ocr_engine is not None:
                for track, plate in self._select_for_ocr(tracks, assignment):
                    read = self._submit_ocr(track, plate, frame, prefetched)
                    if read is not None:            # offline: sincrono e deterministico
                        self._incorporate(track, read, ref)
                        if read.text:
                            reads.append(read)
        timing.ocr_ms = sw_ocr.ms

        with Stopwatch() as sw_analytics:
            self._update_tracks(tracks, assignment, frame, measured=run_detection)
            self._close_ready(ref)
            if self.store is not None:
                self._store_frame_data(frame, vehicles, plates, tracks, assignment, run_detection)
        timing.analytics_ms = sw_analytics.ms

        timing.total_ms = (
            timing.detect_ms + timing.track_ms + timing.ocr_ms + timing.analytics_ms
        )
        t_done = time.monotonic()
        timing.latency_ms = (t_done - frame.timestamp) * 1000.0
        self._last_ref = ref
        result = FrameResult(
            frame=frame, vehicles=vehicles, plates=plates, tracks=tracks,
            reads=reads, timing=timing, dropped_frames=int(dropped),
            frames_skipped=skipped, detection_ran=run_detection,
            extra={"rois_skipped": rois_skipped},
        )
        self.metrics.on_frame(
            total_ms=timing.total_ms, detect_ms=timing.detect_ms, ocr_ms=timing.ocr_ms,
            track_ms=timing.track_ms, dropped=int(dropped), n_active_tracks=len(tracks),
            latency_ms=timing.latency_ms,
        )
        if self.store is not None:
            self.store.add_frame_telemetry(
                result, self.metrics.current_fps(), t0=self._t0, start_t=t_start,
                done_t=t_done, deadline_ms=self._deadline_ms,
                ocr_in_flight=len(self._pending),
            )
        return result

    def _update_tracks(
        self,
        tracks: list[TrackState],
        assignment: dict[int, Detection],
        frame: Frame,
        measured: bool,
    ) -> None:
        """Fusao, melhor recorte, analitica de trafego e publicacao de eventos."""
        ref = FrameRef(frame.index, frame.source_pts, frame.scale)
        # Velocidade, direcao e cruzamentos so com posicao MEDIDA; nos quadros
        # so previstos (detect_every_n) a caixa e extrapolacao do Kalman.
        crossings = (
            self.traffic.update(tracks, frame.index, frame.source_pts, coord_scale=frame.scale)
            if measured else []
        )
        por_id = {t.track_id: t for t in tracks}

        for track in tracks:
            # Contagem de objetos unicos por classe (idempotente por track_id).
            self.metrics.on_track_seen(
                track.track_id, track.object_class.value, track.category.value,
                track.is_vulnerable,
            )
            plate = assignment.get(track.track_id)
            if plate is not None:
                self._maybe_store_best_crop(track, plate, frame)
            if not track.reads:
                continue
            fused = self.voter.fuse(track)
            decisao = self.policy.on_update(track, fused)
            if decisao is not None:
                self._publish(track, decisao, ref)

        for crossing in crossings:
            crossing_track = por_id.get(crossing.track_id)
            if crossing_track is not None and self.store is not None:
                self.store.add_crossing(
                    crossing_track, crossing.line_name, crossing.direction, frame.index, frame.source_pts
                )

    def _end_tracks(self, ended: list[TrackState], ref: FrameRef) -> None:
        """Trilhas que o rastreador encerrou: grava agora ou quando o OCR chegar."""
        for track in ended:
            if not track.reported:
                # Hipotese que nunca chegou a ser confirmada: nao e um objeto.
                self._stats["tracks_discarded_tentative"] += 1
                continue
            self._ending[track.track_id] = track

    def _close_ready(self, ref: FrameRef) -> None:
        for tid in [t for t in self._ending
                    if t not in self._pending or t in self._expired_pending]:
            self._close_track(self._ending.pop(tid), ref)

    def _close_track(self, track: TrackState, ref: FrameRef) -> None:
        """Estado final de uma trilha: ultima fusao, publicacao se cabivel, gravacao."""
        fused: FusedPlate | None = None
        if track.reads:
            fused = self.voter.fuse(track)
            # A mesma politica do laco principal - inclusive no fim do video.
            decisao = self.policy.on_exit(track, fused)
            if decisao is not None:
                self._publish(track, decisao, ref)
        crop_path = ""
        if track.published_text is None and track.reads:
            # Trilha lida mas nunca publicada: o recorte serve para auditar a perda.
            crop_path = self._save_crop(track, ref, "naopublicada")
        if self.store is not None:
            self.store.upsert_track(track, fused, crop_path, scale=ref.scale)
        self.traffic.forget({track.track_id})
        self._last_ocr_frame.pop(track.track_id, None)
        self._published.discard(track.track_id)
        self._stats["tracks_closed"] += 1

    def _maybe_store_best_crop(self, track: TrackState, plate: Detection, frame: Frame) -> None:
        """Guarda o recorte mais nitido da trilha, para auditoria visual.

        A nitidez e medida no recorte reduzido (barato, a cada quadro); o recorte
        guardado e o da maior resolucao disponivel (``_plate_patch``).
        """
        if not self.config.analytics.save_crops:
            return
        patch = crop(frame.image, plate.bbox, pad=self.config.ocr.pad)
        if patch.size == 0:
            return
        score = sharpness(patch) * plate.score
        if score > track.best_crop_score:
            track.best_crop_score = score
            track.best_crop = self._plate_patch(frame, plate.bbox)

    def _save_crop(self, track: TrackState, ref: FrameRef, rotulo: str) -> str:
        if not self.config.analytics.save_crops or track.best_crop is None:
            return ""
        # A sessao entra no nome: IDs de trilha recomecam a cada sessao.
        sessao = self.store.session_id if self.store is not None else "semsessao"
        path = self._crops_dir / f"{sessao}_t{track.track_id}_f{ref.index}_{rotulo or 'x'}.jpg"
        try:
            cv2.imwrite(str(path), track.best_crop)
            return str(path)
        except (OSError, cv2.error) as exc:
            log.debug("Falha ao salvar recorte: %s", exc)
            return ""

    def _publish(self, track: TrackState, decisao: Decision, ref: FrameRef) -> None:
        """Grava uma decisao sobre a placa de uma trilha (ver fusion.publication)."""
        fused = decisao.fused
        anterior = track.published_text
        crop_path = self._save_crop(track, ref, fused.text)
        elapsed_ms = (None if track.first_available_monotonic is None else
                      max(0.0, time.monotonic() - track.first_available_monotonic) * 1000.0)
        if self.store is not None:
            self.store.add_event(track, ref.index, ref.source_pts, fused, crop_path, ref.scale,
                                 decision=decisao.kind, revision=decisao.revision,
                                 decision_elapsed_ms=elapsed_ms)
        # Enqueue de evento critico pode falhar; so confirmar depois de aceito.
        self._published.add(track.track_id)
        self.policy.apply(track, decisao, ref.index, ref.source_pts)
        if self.store is not None:
            # Registro provisorio da trilha (end_reason vazio); a gravacao final,
            # quando a trilha terminar, o substitui com o estado completo.
            self.store.upsert_track(track, fused, crop_path, scale=ref.scale)
        self.metrics.on_event(fused.text, replaces=anterior if decisao.revision else None)
        log.info(
            "PLACA %s (trilha %d, quadro %d, conf %.2f, concordancia %.2f, %d leituras%s)",
            fused.formatted, track.track_id, ref.index, fused.confidence, fused.agreement,
            fused.n_reads, "" if decisao.kind == "stable" else f", {decisao.kind}",
        )

    def _store_frame_data(
        self,
        frame: Frame,
        vehicles: list[Detection],
        plates: list[Detection],
        tracks: list[TrackState],
        assignment: dict[int, Detection],
        measured: bool,
    ) -> None:
        """Deteccoes e estado das trilhas deste quadro (coordenadas originais)."""
        store = self.store
        assert store is not None
        if store.store_detections and measured:
            for det in vehicles:
                store.add_detection(frame.index, frame.source_pts, "object", det, frame.scale)
            for det in plates:
                store.add_detection(frame.index, frame.source_pts, "plate", det, frame.scale)
        if store.store_track_points:
            for track in tracks:
                placa = assignment.get(track.track_id)
                store.add_track_point(
                    track, frame.index, frame.source_pts,
                    placa.bbox if placa is not None else None, frame.scale,
                )

    # -- execucao ----------------------------------------------------------

    def configure_for_source(self, source: Any) -> None:
        """Ajustes que dependem da fonte (prazo padrao = um periodo de quadro)."""
        self._source = source
        if self._deadline_ms is None:
            fps = float(getattr(source, "fps", 0.0) or 0.0)
            if fps > 0:
                self._deadline_ms = 1000.0 / fps

    @property
    def deadline_ms(self) -> float | None:
        return self._deadline_ms

    def run(self, source: VideoSource | ImageFolderSource) -> Iterator[FrameResult]:
        """Itera sobre a fonte, devolvendo o resultado de cada quadro."""
        cfg = self.config.source
        self.configure_for_source(source)
        processed = 0
        for frame in source.frames():
            if self._stop.is_set():
                break
            if frame.index < cfg.start_frame:
                continue
            # O total de descartes entra no resultado ANTES da gravacao da
            # telemetria (antes era atribuido depois, e o banco sempre via 0).
            result = self.process_frame(frame, int(getattr(source, "dropped", 0)))
            yield result
            processed += 1
            if cfg.max_frames and processed >= cfg.max_frames:
                break

    def finalize(self) -> None:
        """Encerra as leituras em voo e grava todas as trilhas ainda vivas.

        Toda trilha confirmada termina gravada em ``tracks``; um *evento* so e
        gerado se a fusao atingir o criterio de publicacao - o mesmo do laco
        principal. Pode ser chamado mais de uma vez.
        """
        ref = self._last_ref or FrameRef(self._frame_index, 0.0, 1.0)
        deadline = time.monotonic() + self.config.ocr.job_timeout_s
        while self._pending and time.monotonic() < deadline:
            self._collect_ocr(ref)
            if self._stop.is_set() or self._pending.keys() <= self._expired_pending:
                break
            time.sleep(0.01)
        if self._pending:
            self._stats["ocr_lost_at_end"] += len(self._pending)
            log.warning("%d leitura(s) de OCR nao terminaram a tempo e foram perdidas.",
                        len(self._pending))
            for future in self._pending.values():
                future.cancel()
            # Manter running futures para stop nao fechar o motor durante read.
            self._expired_pending.update(self._pending)

        for track in self.tracker.close_all("end_of_stream"):
            if track.reported:
                self._close_track(track, ref)
            else:
                self._stats["tracks_discarded_tentative"] += 1
        for tid in list(self._ending):
            self._close_track(self._ending.pop(tid), ref)

    def stats(self) -> dict[str, Any]:
        """Contadores da execucao para ``sessions.stats_json``."""
        m = self.metrics.snapshot()
        return {
            **self._stats,
            "frames_processed": m.frames_processed,
            "frames_dropped": m.frames_dropped,
            "ocr_attempts": m.n_ocr_attempts,
            "ocr_reads_with_text": m.n_reads,
            "events": m.n_events,
            "fallback_active": self._fallback_active,
            "deadline_ms": self._deadline_ms,
            "pipeline_workers": 0,          # serial (o paralelo grava os seus)
        }

    # -- proveniencia ------------------------------------------------------

    def components(self) -> dict[str, Any]:
        """Runtime e dispositivo EFETIVOS de cada componente."""
        def descrever(c: Any) -> dict[str, Any] | None:
            if c is None:
                return None
            try:
                return c.describe()
            except Exception as exc:  # noqa: BLE001
                return {"name": getattr(c, "name", "?"), "error": str(exc)}

        return {
            "object_detector": descrever(self.vehicle_detector),
            "plate_detector": descrever(self.plate_detector),
            "ocr": descrever(self.ocr_engine),
        }

    def device_summary(self) -> str:
        """Resumo curto do dispositivo de cada componente (coluna sessions.device)."""
        partes = []
        comps = self.components()
        for rotulo, info in (
            ("objetos", comps["object_detector"]),
            ("placas", comps["plate_detector"]),
            ("ocr", comps["ocr"]),
        ):
            if info:
                partes.append(f"{rotulo}={info.get('runtime', '?')}@{info.get('device', '?')}")
        return "; ".join(partes) or str(self.device)

    def environment(self) -> dict:
        env = describe_environment()
        env["device_preference"] = str(self.device)
        env["components"] = self.components()
        return env
