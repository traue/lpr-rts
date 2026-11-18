"""Pipeline com deteccao paralela e rastreamento sequencial.

## Por que esta divisao

Os estagios do pipeline tem naturezas opostas:

* **Deteccao e sem estado.** O resultado do quadro *i* nao depende do quadro
  *i-1*. Pode rodar em N quadros simultaneamente.
* **Rastreamento e com estado e ordenado.** O filtro de Kalman e a associacao
  dependem do quadro anterior. Processar fora de ordem corromperia as trilhas.

Dai o desenho: **detectar em paralelo, reordenar, rastrear em serie**. A etapa
sequencial e literalmente a mesma do pipeline serial
(``LPRPipeline.sequential_stage``) - e o que garante resultados identicos.

## OCR antecipado (offline)

O OCR de um recorte tambem e sem estado: so depende do quadro e da caixa da
placa. No perfil que le toda placa em todo quadro (``ocr_every_n: 1``,
``max_reads_published: 0`` - o de avaliacao), cada worker le as placas que
acabou de detectar, com o proprio motor de OCR, e a etapa serial so incorpora
a leitura pronta da mesma deteccao. O resultado e o do serial, bit a bit (os
motores tem a mesma configuracao, inclusive ``ocr.threads``). No modo
so-placa, em tempo real ou com agendamento seletivo, nada e antecipado.

```
                    ┌──▶ detector 1 ──┐
   captura ──▶ fila ├──▶ detector 2 ──┤──▶ reordenacao ──▶ rastreio ──▶ OCR ──▶ analytics
                    └──▶ detector N ──┘   (por sequencia)   (serial)   (pool)
```

## Reordenacao sem buracos

Cada quadro recebe um numero de sequencia **no momento em que um worker o
retira da fila**, sob uma trava. A sequencia nao tem buracos por construcao:
todo quadro retirado produz uma saida, ou uma falha do detector interrompe a
sessao com erro explicito (nao vira um quadro sem deteccoes).

Uma versao anterior reordenava pelo *indice do quadro na fonte*. No modo tempo
real, todo quadro descartado deixava um buraco, e o buffer de reordenacao
esperava o indice que nunca viria ate acumular 32 quadros: a latencia medida
chegou a ~1,6 s com o banco registrando ~90 ms.

## Politica das filas

* Entrada: *drop-oldest* em tempo real (latencia acima de completude) e
  **bloqueante** em offline (perder um quadro invalidaria a avaliacao).
* Saida: sempre **bloqueante**. Descartar depois da deteccao jogaria fora
  trabalho pronto; sob carga a contrapressao chega a fila de entrada, que e o
  lugar certo de descartar.

## Por que threads e nao processos

O tempo e gasto dentro do ONNX Runtime, do PyTorch e do OpenCV, que **liberam o
GIL** durante a computacao nativa. Threads dao paralelismo real e compartilham os
pesos do modelo. Para paralelismo entre *videos*, veja ``core.batch``.

## O compromisso que isto introduz

Deteccao paralela aumenta a **vazao**, nao reduz a **latencia** de um quadro
individual. Num sistema ao vivo com uma camera, prefira N=2 ou 3; para
processar um arquivo o mais rapido possivel, use todos os nucleos.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from ..capture.ringbuffer import BlockingQueue, DropOldestQueue
from ..capture.source import RawFrame
from ..config import AppConfig
from ..detect.base import Detector
from ..ocr.base import OCREngine
from ..utils.logging import get_logger
from .clock import Stopwatch
from .pipeline import LPRPipeline
from .types import Detection, Frame, FrameResult, PlateRead

log = get_logger(__name__)


T = TypeVar("T")
DetectorPair = tuple[Detector | None, Detector | None]
DetectionPayload = tuple[
    Frame, list[Detection], list[Detection], float, bool, int, float, dict[int, PlateRead] | None
]


class ReorderBuffer(Generic[T]):
    """Restaura a ordem dos quadros apos o processamento paralelo.

    Os workers terminam fora de ordem (um quadro com muitos objetos demora mais).
    Este buffer entrega sempre o proximo numero de sequencia esperado, segurando
    os que chegaram adiantados.

    O limite ``max_pending`` evita que um worker travado bloqueie a saida
    indefinidamente: passando dele, o item faltante e considerado perdido e a
    sequencia avanca (e o salto e contado em ``skipped``).
    """

    def __init__(self, max_pending: int = 32) -> None:
        self._heap: list[tuple[int, T]] = []
        self._next = 0
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._skipped = 0

    def push(self, index: int, payload: T) -> list[T]:
        """Insere um resultado e devolve os que ja podem sair, em ordem."""
        with self._lock:
            if index < self._next:
                return []            # chegou tarde demais; ja avancamos
            heapq.heappush(self._heap, (index, payload))
            return self._drain_locked()

    def _drain_locked(self) -> list[T]:
        pronto: list[T] = []
        while self._heap:
            idx, payload = self._heap[0]
            if idx == self._next:
                heapq.heappop(self._heap)
                pronto.append(payload)
                self._next += 1
            elif len(self._heap) > self._max_pending:
                # Buraco na sequencia: pula o indice faltante em vez de travar.
                self._skipped += self._heap[0][0] - self._next
                self._next = self._heap[0][0]
            else:
                break
        return pronto

    def flush(self) -> list[T]:
        """Libera tudo o que restou, em ordem (chamado no encerramento)."""
        with self._lock:
            restante = [payload for _, payload in sorted(self._heap, key=lambda x: x[0])]
            self._heap.clear()
            return restante

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._heap)


def tune_opencv_threads(n_workers: int) -> int:
    """Reparte os nucleos entre os workers e o threading interno do OpenCV.

    Sem isto, cada worker tenta usar todos os nucleos para as operacoes de
    imagem e os N workers passam a disputar os mesmos recursos - o classico
    *oversubscription*, que costuma deixar o conjunto mais lento que a versao
    serial. Retorna quantas threads foram atribuidas ao OpenCV.
    """
    import os

    import cv2

    total = os.cpu_count() or 4
    por_worker = max(1, total // max(1, n_workers))
    cv2.setNumThreads(por_worker)
    return por_worker


class ParallelPipeline(LPRPipeline):
    """``LPRPipeline`` com o estagio de deteccao distribuido entre threads."""

    def __init__(
        self,
        config: AppConfig,
        detector_factory: Callable[[], DetectorPair] | None = None,
        n_workers: int = 2,
        ocr_factory: Callable[[], OCREngine] | None = None,
        **kwargs,
    ) -> None:
        """
        ``detector_factory`` deve devolver ``(detector_objetos, detector_placas)``
        e e chamada uma vez por worker. Fornecer instancias separadas e o caminho
        seguro: nem todo backend garante que uma mesma sessao aceite chamadas
        concorrentes. Sem a fabrica, os detectores recebidos sao compartilhados -
        correto para ONNX Runtime, que e thread-safe em ``run()``.

        ``ocr_factory`` cria um motor de OCR por worker para o OCR antecipado
        (ver docstring do modulo); sem ela, o OCR fica todo na etapa serial.
        """
        super().__init__(config, **kwargs)
        self.n_workers = max(1, n_workers)
        self._detector_factory = detector_factory
        self._worker_detectors: list[DetectorPair] = []
        self._ocr_factory = ocr_factory
        self._worker_ocr: list[OCREngine] = []
        self._antecipa = False                     # sobrevive ao stop (que libera os motores)
        self._prefetched = [0] * self.n_workers   # um contador por worker: sem trava
        # Sem buracos por construcao; o limite so protege contra worker travado.
        self._reorder: ReorderBuffer[DetectionPayload] = ReorderBuffer(
            max_pending=32 if config.source.realtime else 10_000
        )
        self._input: DropOldestQueue[Frame | RawFrame] | BlockingQueue[Frame | RawFrame] = (
            DropOldestQueue(maxsize=self.n_workers * 2) if config.source.realtime
            else BlockingQueue(maxsize=self.n_workers * 2)
        )
        self._output: BlockingQueue[tuple[int, DetectionPayload]] = BlockingQueue(
            maxsize=self.n_workers * 4
        )
        self._dequeue_lock = threading.Lock()
        self._next_seq = 0
        self._source = None
        self._threads: list[threading.Thread] = []
        self._workers_done = threading.Event()
        self._active_workers = 0
        self._worker_lock = threading.Lock()
        self._cv_threads = 0
        self._failure: Exception | None = None
        self._shutdown_lock = threading.Lock()
        self._cleanup_thread: threading.Thread | None = None

    # -- ciclo de vida -----------------------------------------------------

    def start(self) -> ParallelPipeline:
        try:
            super().start()
            self._cv_threads = tune_opencv_threads(self.n_workers)
            if self._detector_factory is not None:
                # Guardar cada instancia imediatamente para rollback parcial.
                for _ in range(self.n_workers):
                    self._worker_detectors.append(self._detector_factory())
            if self.prefetch_enabled and self._ocr_factory is not None:
                for _ in range(self.n_workers):
                    motor = self._ocr_factory()
                    self._worker_ocr.append(motor)
                    # O mesmo limite de correcoes do motor principal: mesma leitura.
                    motor.max_corrections = self.config.ocr.max_grammar_corrections
            self._antecipa = bool(self._worker_ocr)
            aquecer: list[Detector | OCREngine | None] = [self.vehicle_detector, self.plate_detector]
            for par in self._worker_detectors:
                aquecer.extend(par)
            aquecer.extend(self._worker_ocr)
            for componente in aquecer:
                if componente is not None:
                    componente.warmup()
        except Exception:
            self.stop()
            raise
        log.info(
            "Pipeline paralelo: %d workers de deteccao, %d threads de OpenCV cada%s",
            self.n_workers, self._cv_threads,
            ", OCR antecipado" if self._antecipa else "",
        )
        return self

    @property
    def prefetch_enabled(self) -> bool:
        """OCR antecipado nos workers: offline, com veiculos, OCR e agendamento que le tudo."""
        return (not self.config.source.realtime and self.ocr_engine is not None
                and self.reads_every_plate and not self._plate_only_mode)

    def request_stop(self) -> None:
        super().request_stop()
        self._input.close()
        self._output.close()

    def stop(self) -> None:
        self.request_stop()
        # O mesmo lock protege registrar/iniciar threads: stop nunca pode
        # liberar um modelo usado por uma thread ainda nao registrada.
        with self._shutdown_lock:
            if self._ocr_pool is not None:
                self._ocr_pool.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + 2.0
            for thread in self._threads:
                if thread.is_alive():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self._threads):
                if self._cleanup_thread is None:
                    self._cleanup_thread = threading.Thread(
                        target=self._deferred_stop, args=(tuple(self._threads),),
                        name="parallel-cleanup", daemon=True,
                    )
                    self._cleanup_thread.start()
                raise TimeoutError("Workers ainda em execucao apos cancelamento")
            self._finish_stop()

    def _deferred_stop(self, workers: tuple[threading.Thread, ...]) -> None:
        """Uma unica espera por pipeline; nao exige outra chamada do cliente."""
        for thread in workers:
            if thread.ident is not None:
                thread.join()
        with self._shutdown_lock:
            try:
                self._finish_stop()
            except Exception as exc:  # OCR em voo tem seu proprio callback de liberacao
                log.warning("Encerramento adiado do pipeline: %s", exc)

    def _finish_stop(self) -> None:
        """Chamado sob _shutdown_lock somente depois da saida dos workers."""
        self._threads.clear()
        for pair in self._worker_detectors:
            for detector in pair:
                self._close_component(detector)
        self._worker_detectors.clear()
        for motor in self._worker_ocr:
            self._close_component(motor)
        self._worker_ocr.clear()
        super().stop()

    def _fail(self, exc: Exception) -> None:
        with self._worker_lock:
            if self._failure is None:
                self._failure = exc
        self.request_stop()

    # -- estagio paralelo --------------------------------------------------

    def _detectors_for(self, worker_id: int) -> DetectorPair:
        if self._worker_detectors:
            return self._worker_detectors[worker_id]
        return (self.vehicle_detector, self.plate_detector)

    def _detect_worker(self, worker_id: int) -> None:
        """Retira quadros em ordem; falhas sao entregues ao consumidor."""
        try:
            detectores = self._detectors_for(worker_id)
            while not self._stop.is_set():
                with self._dequeue_lock:
                    frame = self._input.get(timeout=0.1)
                    if frame is not None:
                        seq = self._next_seq
                        self._next_seq += 1
                        run = self.should_detect(frame.index)
                if frame is None:
                    if self._input.closed and len(self._input) == 0:
                        break
                    continue
                if isinstance(frame, RawFrame):
                    # Reducao/rotacao fora da thread de leitura: em paralelo, mesmos pixels.
                    frame = self._source.prepare(frame)
                inicio = time.monotonic()
                with Stopwatch() as sw:
                    objetos, placas, puladas = self.detect_stage(frame.image, run, detectores)
                lidas = None
                if self._worker_ocr and run and placas:
                    lidas = self.prefetch_ocr(frame, placas, self._worker_ocr[worker_id])
                    self._prefetched[worker_id] += len(lidas)
                payload = (frame, objetos, placas, sw.ms, run, puladas, inicio, lidas)
                self._output.put((seq, payload))
        except Exception as exc:
            self._fail(exc)
        finally:
            with self._worker_lock:
                self._active_workers -= 1
                if self._active_workers <= 0:
                    # Nenhuma instancia dedicada esta mais em inferencia. A
                    # liberacao tambem acontece apos stop ter reportado timeout.
                    base_ids = {id(self.vehicle_detector), id(self.plate_detector)}
                    for pair in self._worker_detectors:
                        for detector in pair:
                            if id(detector) not in base_ids:
                                self._close_component(detector)
                    for motor in self._worker_ocr:
                        self._close_component(motor)
                    self._workers_done.set()
                    self._output.close()

    def _feeder(self, source) -> None:
        """Le a fonte; fecha a entrada mesmo quando ela falha ou e cancelada.

        Arquivo offline: so decodifica (``raw_frames``) e os workers preparam cada
        quadro. No video 4K vertical a leitura caia de ~8 para ~1,3 ms por quadro;
        com tudo nesta thread, ela limitava o paralelo abaixo do serial.
        """
        cfg = self.config.source
        enviados = 0
        try:
            crus = bool(getattr(source, "supports_raw_frames", False)) and not cfg.realtime
            for frame in (source.raw_frames() if crus else source.frames()):
                if self._stop.is_set():
                    break
                if frame.index < cfg.start_frame:
                    continue
                self._input.put(frame)
                enviados += 1
                if cfg.max_frames and enviados >= cfg.max_frames:
                    break
        except Exception as exc:
            self._fail(exc)
        finally:
            self._input.close()

    # -- execucao ----------------------------------------------------------

    def run(self, source) -> Iterator[FrameResult]:
        """Itera sobre os resultados, ja reordenados, do pipeline paralelo."""
        self.configure_for_source(source)
        self._source = source
        with self._shutdown_lock:
            if self._stop.is_set():
                return
            self._workers_done.clear()
            with self._worker_lock:
                self._active_workers = self.n_workers
            feeder = threading.Thread(target=self._feeder, args=(source,), name="feeder", daemon=True)
            self._threads.append(feeder)
            feeder.start()
            for i in range(self.n_workers):
                thread = threading.Thread(target=self._detect_worker, args=(i,),
                                          name=f"detect-{i}", daemon=True)
                self._threads.append(thread)
                thread.start()

        while not self._stop.is_set():
            item = self._output.get(timeout=0.1)
            if item is None:
                if self._workers_done.is_set() and len(self._output) == 0:
                    break
                continue
            seq, payload = item
            for pronto in self._reorder.push(seq, payload):
                yield self._finish_frame(*pronto)

        if self._failure is not None:
            raise RuntimeError("Falha no pipeline paralelo: " + str(self._failure)) from self._failure
        if not self._stop.is_set():
            for pronto in self._reorder.flush():
                yield self._finish_frame(*pronto)

    def _dropped_total(self) -> int:
        return (
            int(getattr(self._source, "dropped", 0))
            + self._input.dropped
            + self._reorder.skipped
        )

    def _finish_frame(
        self,
        frame: Frame,
        objetos: list[Detection],
        placas: list[Detection],
        detect_ms: float,
        run_detection: bool = True,
        rois_skipped: int = 0,
        t_start: float | None = None,
        prefetched: dict[int, PlateRead] | None = None,
    ) -> FrameResult:
        """Etapa sequencial - o mesmo codigo do pipeline serial, na thread consumidora."""
        return self.sequential_stage(
            frame, objetos, placas, detect_ms, run_detection, rois_skipped,
            self._dropped_total(), t_start, prefetched,
        )

    def stats(self) -> dict[str, Any]:
        """Contadores do serial mais os do paralelo (workers, leituras antecipadas)."""
        dados = super().stats()
        dados["pipeline_workers"] = self.n_workers
        dados["ocr_prefetched"] = sum(self._prefetched)
        return dados

    # -- diagnostico -------------------------------------------------------

    def parallel_stats(self) -> dict[str, int | float]:
        """Metricas especificas do paralelismo, para diagnostico."""
        return {
            "workers": self.n_workers,
            "opencv_threads_por_worker": self._cv_threads,
            "quadros_descartados_fonte": int(getattr(self._source, "dropped", 0)),
            "quadros_descartados_entrada": self._input.dropped,
            "quadros_pulados_reordenacao": self._reorder.skipped,
            "pendentes_na_reordenacao": self._reorder.pending,
            "ocr_antecipado": self._antecipa,
            "leituras_antecipadas": sum(self._prefetched),
            "leituras_antecipadas_usadas": self._stats["ocr_prefetch_used"],
        }


def choose_worker_count(target: str = "auto", ocr_antecipado: bool = False) -> int:
    """Numero de workers de deteccao segundo o objetivo.

    ``latencia`` privilegia responsividade (poucos workers, menos reordenacao);
    ``vazao`` usa todos os nucleos; ``auto`` fica no meio, deixando folga para a
    interface, o OCR e a escrita no banco. Com OCR antecipado o trabalho de cada
    worker passa a ser sobretudo CPU, e o ``auto`` sobe para metade dos nucleos.

    Medido no Apple M5 Pro (15 nucleos, deteccao no CoreML): perfil padrao em 4K
    vertical 50 q/s serial, 78/83/87 com 2/3/4 workers; perfil de avaliacao 20 q/s
    serial, 59 e 61-63 com 6 e 8, 53 com 10.
    """
    import os

    total = os.cpu_count() or 4
    alvo = (target or "auto").lower()
    if alvo == "latencia":
        return 1 if total <= 2 else 2
    if alvo == "vazao":
        return max(1, total)
    if ocr_antecipado:
        return max(1, min(8, total // 2))
    return max(1, min(4, total - 1))


def auto_workers(config: AppConfig) -> int:
    """Contagem automatica para esta configuracao (sobe se o OCR for antecipado)."""
    cfg_ocr = config.ocr
    antecipa = (config.detector.detect_vehicles and cfg_ocr.ocr_every_n <= 1
                and cfg_ocr.max_reads_published == 0)
    return choose_worker_count("auto", ocr_antecipado=antecipa)


def resolve_workers(config: AppConfig) -> int:
    """Workers efetivos de ``detector.workers`` (0 = pipeline serial).

    O automatico fica serial em tempo real: com a maquina acompanhando a camera
    (medido: 30 q/s, sem descarte, nos dois videos 4K), a reordenacao so
    acrescentaria latencia.
    """
    n = config.detector.workers
    if n >= 0:
        return n
    return 0 if config.source.realtime else auto_workers(config)
