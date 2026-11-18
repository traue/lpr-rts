"""Robustez em execucao: falhas, cancelamento, prazos e limites (sem modelos neurais)."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest

from lprts.capture.source import VideoSource
from lprts.config import AppConfig
from lprts.core.parallel import ParallelPipeline
from lprts.core.pipeline import LPRPipeline
from lprts.core.types import Detection, Frame, PlateRead
from lprts.track.bytetrack import ByteTracker


def config():
    cfg = AppConfig()
    cfg.detector.device = "cpu"
    cfg.detector.detect_vehicles = False
    cfg.analytics.save_crops = False
    cfg.tracker.min_hits = 1
    cfg.fusion.min_reads = 1
    cfg.ocr.min_plate_width = 10
    return cfg


def frame(index):
    return Frame(index, np.zeros((100, 480, 3), dtype=np.uint8), time.monotonic(), index / 30)


def plate(x=10):
    return Detection((x, 10, x + 80, 40), .9, "plate")


def test_trilha_expirada_nao_transfere_identidade_publicada():
    tracker = ByteTracker(min_hits=1, max_age=30)
    old = tracker.update([plate()], 0, 0)[0]
    old.published_text = "ABC1D23"
    new = tracker.update([plate()], 100, 100 / 30)[0]
    assert new.track_id != old.track_id
    assert new.published_text is None
    assert tracker.pop_removed() == [old]


def test_atribuicao_do_mota_acha_todos_os_pares_admissiveis():
    from lprts.track.assignment import gated_assignment
    rows, cols = gated_assignment(np.array([[.8, .4], [.4, .075]]), .3)
    assert set(zip(rows, cols, strict=True)) == {(0, 1), (1, 0)}
    rows, cols = gated_assignment(np.array([[np.nan, .5], [-np.inf, .4]]), .3)
    assert list(zip(rows, cols, strict=True)) == [(0, 1)]


def test_leitura_pronta_no_quadro_da_remocao_e_incorporada():
    pipe = LPRPipeline(config())
    track = pipe.sequential_stage(frame(0), [], [plate()], 0, True).tracks[0]
    future = Future()
    future.set_result(PlateRead("ABC1D23", .99, valid_syntax=True, frame_index=0))
    pipe._pending[track.track_id] = future
    pipe.sequential_stage(frame(100), [], [], 0, True)
    assert pipe.stats()["orphan_reads"] == 0
    assert track.n_reads_total == 1
    assert track.published_text == "ABC1D23"
    assert pipe.stats()["tracks_closed"] == 1


class CapturaQueFalha:
    def read(self):
        raise RuntimeError("decoder failed")

    def release(self):
        pass


def test_excecao_da_captura_chega_ao_consumidor_e_fecha_a_fila():
    source = VideoSource(0, realtime=True)
    source._cap = CapturaQueFalha()
    errors = []

    def consume():
        try:
            list(source.frames())
        except Exception as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(1)
    try:
        assert not consumer.is_alive()
        assert len(errors) == 1 and "decoder failed" in str(errors[0])
        assert source._queue.closed
    finally:
        source.close()
        consumer.join(2)


class FonteQuebrada:
    fps = 30
    dropped = 0

    def frames(self):
        raise RuntimeError("source failed")
        yield


def test_excecao_do_alimentador_paralelo_chega_ao_consumidor():
    pipe = ParallelPipeline(config(), n_workers=2).start()
    errors = []

    def consume():
        try:
            list(pipe.run(FonteQuebrada()))
        except Exception as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(1)
    try:
        assert not consumer.is_alive()
        assert len(errors) == 1 and "source failed" in str(errors[0])
        assert pipe._input.closed
    finally:
        pipe.stop()
        consumer.join(2)


class DetectorContador:
    def __init__(self):
        self.closed = 0

    def warmup(self):
        pass

    def close(self):
        self.closed += 1


def test_cada_detector_paralelo_e_fechado_uma_vez():
    base = DetectorContador()
    workers = []

    def factory():
        detector = DetectorContador()
        workers.append(detector)
        return None, detector

    pipe = ParallelPipeline(config(), detector_factory=factory, n_workers=2,
                            plate_detector=base).start()
    pipe.stop()
    pipe.stop()
    assert base.closed == 1
    assert [det.closed for det in workers] == [1, 1]
    assert pipe._worker_detectors == []


def test_partida_parcial_do_paralelo_libera_os_modelos():
    base, first = DetectorContador(), DetectorContador()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("model failed")
        return None, first

    pipe = ParallelPipeline(config(), detector_factory=factory, n_workers=2,
                            plate_detector=base)
    with pytest.raises(RuntimeError, match="model failed"):
        pipe.start()
    assert base.closed == first.closed == 1


class OCRLento:
    name = "slow"

    def __init__(self, gate):
        self.gate = gate

    def read(self, image):
        assert self.gate.wait(3), "test OCR gate was not released"
        return PlateRead("ABC1D23", .99, valid_syntax=True)

    def close(self):
        pass


def test_sobrecarga_de_ocr_tem_teto_e_vencidos_ocupam_capacidade(monkeypatch):
    cfg = config()
    cfg.source.realtime = True
    cfg.tracker.max_age = 1
    cfg.ocr.workers = 1
    cfg.ocr.ocr_every_n = 1
    cfg.ocr.max_pending_jobs = 4
    cfg.ocr.job_timeout_s = .5
    gate = threading.Event()
    pipe = LPRPipeline(cfg, ocr_engine=OCRLento(gate)).start()
    try:
        for i in range(120):
            pipe.sequential_stage(frame(i), [], [plate(10 + 120 * (i % 3))], 0, True)
        assert len(pipe._pending) <= 4
        assert len(pipe._ending) <= 4
        assert pipe.stats()["ocr_backpressure"] > 0
        # Envelhecer a submissao nao libera capacidade enquanto os workers retem os recortes.
        pipe._pending_since = {tid: time.monotonic() - 1 for tid in pipe._pending}
        for i in range(120, 140):
            pipe.sequential_stage(frame(i), [], [plate(10 + 120 * (i % 3))], 0, True)
        assert len(pipe._pending) <= 4
        assert not pipe._ending
        assert pipe.stats()["ocr_expired"] == 4
    finally:
        gate.set()
        pipe.finalize()
        pipe.stop()


def test_decisao_grava_o_relogio_ate_no_fim_do_video(monkeypatch):
    cfg = config()
    pipe = LPRPipeline(cfg)
    first = frame(0)
    track = pipe.sequential_stage(first, [], [plate()], 0, True).tracks[0]
    future = Future()
    future.set_result(PlateRead("ABC1D23", .99, valid_syntax=True, frame_index=0))
    pipe._pending[track.track_id] = future
    recorded = []

    class Store:
        def add_event(self, *args, **kwargs):
            recorded.append(kwargs)

        def add_read(self, *args, **kwargs):
            pass

        def upsert_track(self, *args, **kwargs):
            pass

    pipe.store = Store()
    monkeypatch.setattr("lprts.core.pipeline.time.monotonic", lambda: first.timestamp + .221)
    pipe.finalize()
    assert len(recorded) == 1
    assert recorded[0]["decision_elapsed_ms"] == pytest.approx(221)
    assert track.published_time_s == 0


def test_evento_recusado_nao_marca_a_trilha_como_publicada():
    pipe = LPRPipeline(config())
    track = pipe.sequential_stage(frame(0), [], [plate()], 0, True).tracks[0]
    pipe.voter.add_read(track, PlateRead("ABC1D23", .99, valid_syntax=True))

    class Store:
        def add_event(self, *args, **kwargs):
            raise RuntimeError("storage full")

    pipe.store = Store()
    decision = pipe.policy.on_update(track, pipe.voter.fuse(track))
    with pytest.raises(RuntimeError, match="storage full"):
        pipe._publish(track, decision, pipe._last_ref)
    assert track.published_text is None
    assert track.track_id not in pipe._published
    assert pipe.metrics.snapshot().n_events == 0


def test_stream_ao_vivo_abre_com_prazos_nativos(monkeypatch):
    calls = []
    monkeypatch.setattr("lprts.capture.source.cv2.VideoCapture",
                        lambda *args: calls.append(args) or object())
    source = VideoSource("rtsp://example.test/live", open_timeout_ms=321, read_timeout_ms=654)
    source._abrir(source.uri)
    import cv2
    assert calls == [(source.uri, cv2.CAP_FFMPEG,
                      [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 321, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 654])]


def test_cancelar_a_fonte_libera_o_consumidor_sem_soltar_a_captura():
    entered, release = threading.Event(), threading.Event()
    closed = []

    class Capture:
        def read(self):
            entered.set()
            release.wait(3)
            return False, None

        def release(self):
            assert release.is_set(), "must not release native capture during read"
            closed.append(True)

    source = VideoSource(0, realtime=True, reconnect_attempts=0)
    source._cap = Capture()
    consumer = threading.Thread(target=lambda: list(source.frames()), daemon=True)
    consumer.start()
    assert entered.wait(1)
    try:
        source.cancel()
        consumer.join(1)
        assert not consumer.is_alive()
        assert not closed
    finally:
        release.set()
        source.close()
    assert closed == [True]


def test_cancelar_o_paralelo_acorda_quem_espera_e_preserva_modelos():
    entered, release = threading.Event(), threading.Event()

    class BlockingDetector(DetectorContador):
        def detect(self, image):
            entered.set()
            assert release.wait(3)
            assert self.closed == 0
            return []

    class Source:
        fps = 30
        dropped = 0

        def frames(self):
            yield frame(0)

    detector = BlockingDetector()
    pipe = ParallelPipeline(config(), n_workers=1, plate_detector=detector).start()
    consumer = threading.Thread(target=lambda: list(pipe.run(Source())), daemon=True)
    consumer.start()
    assert entered.wait(1)
    try:
        pipe.request_stop()
        consumer.join(1)
        assert not consumer.is_alive()
        assert detector.closed == 0
    finally:
        release.set()
        pipe.stop()
    assert detector.closed == 1


def test_vigia_da_captura_relata_travamento_sem_liberar_em_paralelo():
    entered, release = threading.Event(), threading.Event()

    class Capture:
        def read(self):
            entered.set()
            assert release.wait(3)
            return False, None

        def release(self):
            assert release.is_set()

    source = VideoSource(0, realtime=True, read_timeout_ms=20, reconnect_attempts=0)
    source._cap = Capture()
    errors = []

    def consume():
        try:
            list(source.frames())
        except Exception as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert entered.wait(1)
    consumer.join(1)
    try:
        assert not consumer.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    finally:
        release.set()
        source.close()


def test_identidades_recomecam_quando_o_relogio_da_fonte_volta():
    tracker = ByteTracker(min_hits=1)
    old = tracker.update([plate()], 100, 10)[0]
    new = tracker.update([plate()], 0, 0)[0]
    assert old.track_id != new.track_id
    assert tracker.pop_removed() == [old]
    assert old.end_reason == "reset"


def test_ocr_vencido_na_fila_nao_e_inferido(monkeypatch):
    class OCR:
        name = "unused"

        def read(self, patch):
            pytest.fail("stale queued OCR should never execute inference")

    cfg = config()
    cfg.source.realtime = True
    cfg.ocr.job_timeout_s = .1
    pipe = LPRPipeline(cfg, ocr_engine=OCR())
    pipe._ocr_pool = object()  # marca o caminho assincrono
    with pytest.raises(TimeoutError, match="expirou"):
        pipe._run_ocr(np.zeros((30, 80, 3)), plate().bbox, 0, 0, time.monotonic() - 1)


def test_ocr_em_voo_no_encerramento_libera_o_motor_depois():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class BlockingOCR:
        name = "blocking"

        def read(self, patch):
            entered.set()
            assert release.wait(3)
            assert not closed.is_set()
            return PlateRead("ABC1D23", .99, valid_syntax=True)

        def close(self):
            assert release.is_set()
            closed.set()

    cfg = config()
    cfg.source.realtime = True
    cfg.ocr.job_timeout_s = .01
    detector = DetectorContador()
    pipe = LPRPipeline(cfg, plate_detector=detector, ocr_engine=BlockingOCR()).start()
    pipe.sequential_stage(frame(0), [], [plate()], 0, True)
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match="OCR ainda"):
            pipe.stop()
        assert detector.closed == 1
        assert not closed.is_set()
    finally:
        release.set()
    assert closed.wait(1)
    pipe.stop()


def test_reconexoes_esgotadas_sao_falha_da_sessao():
    class DisconnectedCapture:
        def read(self):
            return False, None

        def release(self):
            pass

    source = VideoSource(0, realtime=True, reconnect_attempts=0)
    source._cap = DisconnectedCapture()
    try:
        with pytest.raises(RuntimeError, match="reconex"):
            list(source.frames())
    finally:
        source.close()


def test_contexto_do_pipeline_libera_modelos_se_finalizar_falha(monkeypatch):
    detector = DetectorContador()
    pipe = LPRPipeline(config(), plate_detector=detector)

    def fail_finalize():
        raise RuntimeError("storage failed during finalization")

    monkeypatch.setattr(pipe, "finalize", fail_finalize)
    with pytest.raises(RuntimeError, match="storage failed"), pipe:
        pass
    assert detector.closed == 1


@pytest.mark.parametrize("dedicated", [False, True])
def test_paralelo_termina_a_limpeza_quando_o_worker_nativo_retorna(dedicated):
    entered, release = threading.Event(), threading.Event()
    detector_closed, ocr_closed = threading.Event(), threading.Event()

    class BlockingDetector(DetectorContador):
        def detect(self, image):
            entered.set()
            assert release.wait(10)
            assert self.closed == 0
            return []

        def close(self):
            super().close()
            detector_closed.set()

    class OCR(DetectorContador):
        name = "ocr"

        def close(self):
            super().close()
            ocr_closed.set()

    class Source:
        fps = 30
        dropped = 0

        def frames(self):
            yield frame(0)

    cfg = config()
    cfg.source.realtime = True
    detector, base, ocr = BlockingDetector(), DetectorContador(), OCR()
    pipe = ParallelPipeline(
        cfg, n_workers=1, plate_detector=base if dedicated else detector, ocr_engine=ocr,
        detector_factory=(lambda: (None, detector)) if dedicated else None,
    ).start()
    consumer = threading.Thread(target=lambda: list(pipe.run(Source())), daemon=True)
    consumer.start()
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match="Workers ainda"):
            pipe.stop()
        cleanup = getattr(pipe, "_cleanup_thread", None)
        assert cleanup is not None and cleanup.is_alive()
        with pytest.raises(TimeoutError, match="Workers ainda"):
            pipe.stop()
        assert pipe._cleanup_thread is cleanup
        assert not detector_closed.is_set()
    finally:
        release.set()
        consumer.join(1)
    # Nenhum segundo stop e necessario depois que a inferencia nativa retorna.
    assert detector_closed.wait(1)
    assert ocr_closed.wait(1)
    cleanup.join(1)
    assert not cleanup.is_alive()
    assert detector.closed == ocr.closed == 1
    if dedicated:
        assert base.closed == 1
