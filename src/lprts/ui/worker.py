"""Thread de processamento que alimenta a interface grafica.

A interface roda no laco de eventos do Qt; o pipeline roda aqui, numa thread
separada. Os quadros anotados atravessam por sinais Qt, que sao thread-safe.

Detalhe importante de tempo real: a interface consome o **quadro mais recente**,
nunca uma fila acumulada. Se a renderizacao atrasar, o correto e pular quadros -
uma imagem antiga na tela e pior que uma imagem perdida.
"""

from __future__ import annotations

import time
from threading import Event

from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from ..analytics.store import ResultStore
from ..capture.source import ImageFolderSource, VideoSource
from ..config import AppConfig
from ..core.pipeline import LPRPipeline
from ..core.types import FrameResult
from ..factory import (
    build_configured_pipeline,
    build_source,
    build_store,
    check_calibration,
    clear_previous_results,
    close_session,
    open_session,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


def _erro_de_configuracao(exc: BaseException, iniciando: bool) -> bool:
    """True para falhas esperadas, que o usuario resolve sem depurar o codigo."""
    from ..analytics.store import SchemaVersionError
    from ..config import ConfigError
    from ..ocr.engines import EngineUnavailableError

    if isinstance(exc, (SchemaVersionError, ConfigError, EngineUnavailableError,
                        FileNotFoundError, NotADirectoryError)):
        return True
    # Na montagem, ValueError/RuntimeError vem de validacoes (calibracao em outra
    # resolucao, fonte que nao abre); durante o processamento, sao defeitos.
    return iniciando and isinstance(exc, (ValueError, RuntimeError))


class PipelineWorker(QThread):
    """Executa o pipeline e emite os resultados para a interface."""

    frame_ready = Signal(object)        # FrameResult
    plate_found = Signal(str, float, int, str)   # texto, confianca, track_id, classe
    metrics_ready = Signal(object)      # RuntimeMetrics
    status = Signal(str)
    warning = Signal(str)              # aviso persistente, visivel na barra de estado
    detail = Signal(str)               # informacao para a barra de estado (ex.: limpeza)
    failed = Signal(str)
    finished_run = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.pipeline: LPRPipeline | None = None
        self.store: ResultStore | None = None
        self._source: VideoSource | ImageFolderSource | None = None
        self._cancelled = Event()
        self._running = False
        self._paused = False
        self._mutex = QMutex()
        self._seen_plates: dict[int, str] = {}      # trilha -> ultimo texto mostrado
        self._last_emit = 0.0
        self._fallback_avisado = False

    # -- controle ----------------------------------------------------------

    def stop(self) -> None:
        self._cancelled.set()
        with QMutexLocker(self._mutex):
            self._running = False
        if self.pipeline is not None:
            self.pipeline.request_stop()
        if self._source is not None:
            self._source.cancel()

    def toggle_pause(self) -> bool:
        with QMutexLocker(self._mutex):
            self._paused = not self._paused
            return self._paused

    @property
    def paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    @property
    def running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    # -- execucao ----------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - laco principal, mantido linear de proposito
        self._motivo = "fim_da_fonte"
        iniciando = True
        try:
            if self._cancelled.is_set():
                self._motivo = "interrompido"
                return
            self.status.emit("Abrindo a fonte de video...")
            self._source = build_source(self.config)
            check_calibration(self.config, self._source)
            self.status.emit("Carregando modelos...")
            self.pipeline = build_configured_pipeline(self.config)
            if self._cancelled.is_set():
                self._motivo = "interrompido"
                return
            self.pipeline.configure_for_source(self._source)
            # Modelos carregados e aquecidos: falhas de inicializacao preservam o historico.
            if self._cancelled.is_set():
                self._motivo = "interrompido"
                return
            limpeza = clear_previous_results(self.config)
            if limpeza is not None:
                self.detail.emit(limpeza.describe())
            self.store = build_store(self.config)
            self.pipeline.store = self.store
            open_session(self.store, self.config, self._source, self.pipeline, notes="interface")

            with QMutexLocker(self._mutex):
                self._running = not self._cancelled.is_set()
            if not self.running:
                self._motivo = "interrompido"
                return
            self.status.emit("Executando")
            iniciando = False

            for result in self.pipeline.run(self._source):
                if not self.running:
                    self._motivo = "interrompido"
                    break
                while self.paused and self.running:
                    self.msleep(40)
                self._emit(result)

        except Exception as exc:  # noqa: BLE001 - a interface precisa reportar, nao morrer
            self._motivo = "erro"
            if _erro_de_configuracao(exc, iniciando):
                # Condicao esperada, com solucao conhecida (banco antigo, motor
                # ausente, fonte inexistente...): a mensagem basta. Um traceback
                # aqui parece defeito do programa e esconde a instrucao.
                log.error("Nao foi possivel iniciar: %s", exc)
            else:
                log.exception("Falha no pipeline")
            self.failed.emit(str(exc))
        finally:
            self._teardown()
            self.finished_run.emit()

    def _emit(self, result: FrameResult) -> None:
        # A interface desenha em outra thread enquanto o pipeline continua
        # alterando as trilhas: entregar uma copia evita ler um objeto no meio
        # de uma atualizacao.
        self.frame_ready.emit(result.snapshot())

        # O fallback automatico muda o que esta sendo rastreado. Precisa aparecer
        # na interface: e a diferenca entre "o video nao tem placas" e "o detector
        # de objetos nao serve para este video".
        if (
            self.pipeline is not None
            and self.pipeline.fallback_active
            and not self._fallback_avisado
        ):
            self._fallback_avisado = True
            self.warning.emit(self.pipeline.fallback_reason)

        for track in result.tracks:
            # A decisao publicada (que pode ser revisada, fusion.publish_policy:
            # revise); antes dela - ou com on_exit, que so decide na saida -, a
            # fusao estavel corrente.
            texto = track.published_text or (track.plate_text if track.plate_stable else None)
            if texto and self._seen_plates.get(track.track_id) != texto:
                self._seen_plates[track.track_id] = texto
                self.plate_found.emit(
                    texto, track.plate_confidence, track.track_id, track.object_class.value,
                )

        # Metricas a ~5 Hz: atualizar tabelas a cada quadro consome mais CPU na
        # interface do que o proprio pipeline.
        now = time.monotonic()
        if self.pipeline is not None and now - self._last_emit > 0.2:
            self._last_emit = now
            self.metrics_ready.emit(self.pipeline.metrics.snapshot())

    def _teardown(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        actions = []
        if self.pipeline is not None:
            actions.append(self.pipeline.finalize)
        if self.store is not None and self.pipeline is not None:
            store, pipeline, source = self.store, self.pipeline, self._source

            def finish_session() -> None:
                close_session(store, pipeline, source, getattr(self, "_motivo", ""))

            actions.append(finish_session)
        if self.pipeline is not None:
            actions.append(self.pipeline.stop)
        if self.store is not None:
            actions.append(self.store.close)
        if self._source is not None:
            actions.append(self._source.close)
        for action in actions:
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                self._motivo = "erro"
                log.exception("Erro ao encerrar o pipeline")
                self.failed.emit(str(exc))
        self.status.emit("Parado")
