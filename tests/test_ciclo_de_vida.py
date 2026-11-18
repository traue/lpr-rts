"""Cancelamento e falhas no encerramento nao podem parecer partida ou execucao bem-sucedida."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lprts.config import AppConfig


def test_cancelar_ao_configurar_a_fonte_preserva_o_historico(monkeypatch):
    pytest.importorskip("PySide6")
    import lprts.ui.worker as module

    worker = module.PipelineWorker(AppConfig())
    source = SimpleNamespace(close=Mock(), cancel=Mock())
    pipeline = SimpleNamespace(
        stop=Mock(), request_stop=Mock(), finalize=Mock(),
        configure_for_source=lambda _: worker.stop(),
    )
    clear = Mock(return_value=None)
    monkeypatch.setattr(module, "build_source", lambda _: source)
    monkeypatch.setattr(module, "check_calibration", Mock())
    monkeypatch.setattr(module, "build_configured_pipeline", lambda _: pipeline)
    monkeypatch.setattr(module, "clear_previous_results", clear)
    monkeypatch.setattr(module, "build_store", Mock())
    monkeypatch.setattr(module, "open_session", Mock())
    monkeypatch.setattr(module, "close_session", Mock())
    worker.run()
    clear.assert_not_called()
    source.close.assert_called_once()
    pipeline.stop.assert_called_once()


def test_falha_no_lote_grava_a_sessao_encerrada_com_erro(tmp_path, monkeypatch):
    import lprts.factory as factory
    from lprts.analytics.store import ResultStore
    from lprts.core.batch import process_one_video

    db = tmp_path / "failed.db"
    store = ResultStore(db, async_writes=False).open()
    source = SimpleNamespace(close=Mock())
    pipeline = SimpleNamespace(
        run=Mock(side_effect=RuntimeError("decoder failed")), stop=Mock(),
        configure_for_source=Mock(), environment=lambda: {}, device_summary=lambda: "test",
        vehicle_detector=None, plate_detector=None, ocr_engine=None, deadline_ms=None,
        stats=lambda: {},
    )
    monkeypatch.setattr(factory, "build_source", lambda _: source)
    monkeypatch.setattr(factory, "build_store", lambda _: store)
    monkeypatch.setattr(factory, "build_pipeline", lambda *_: pipeline)
    result = process_one_video("test.mp4", AppConfig().to_dict(), db_path=str(db))
    assert not result.ok
    assert "decoder failed" in result.error
    with sqlite3.connect(db) as conn:
        finished, reason = conn.execute("SELECT finished_at, end_reason FROM sessions").fetchone()
    assert finished is not None
    assert reason == "erro"
    pipeline.stop.assert_called_once()
    source.close.assert_called_once()


@pytest.mark.parametrize("run_fails", [True, False])
def test_falha_ao_encerrar_no_lote_mantem_o_erro_original(monkeypatch, run_fails):
    import lprts.factory as factory
    from lprts.core.batch import process_one_video

    source = SimpleNamespace(close=Mock())
    store = SimpleNamespace(close=Mock(), session_id="test")
    pipeline = SimpleNamespace(run=Mock(return_value=iter(())), finalize=Mock(), stop=Mock())
    if run_fails:
        pipeline.run.side_effect = RuntimeError("decoder failed")
    finish = Mock(side_effect=RuntimeError("finish failed"))
    monkeypatch.setattr(factory, "build_source", lambda _: source)
    monkeypatch.setattr(factory, "build_store", lambda _: store)
    monkeypatch.setattr(factory, "build_pipeline", lambda *_: pipeline)
    monkeypatch.setattr(factory, "open_session", Mock())
    monkeypatch.setattr(factory, "close_session", finish)
    result = process_one_video("test.mp4", AppConfig().to_dict(), db_path="unused.db")
    assert not result.ok
    assert "finish failed" in result.error
    if run_fails:
        assert "decoder failed" in result.error
    finish.assert_called_once()
    pipeline.stop.assert_called_once()
    store.close.assert_called_once()
    source.close.assert_called_once()


def test_interromper_a_partida_do_paralelo_fecha_os_modelos_criados(monkeypatch):
    import lprts.factory as factory

    detectors = []

    def build_vehicle(_):
        model = Mock()
        model.warmup.side_effect = KeyboardInterrupt()
        detectors.append(model)
        return model

    def build_plate(_):
        model = Mock()
        detectors.append(model)
        return model

    ocr = Mock()
    monkeypatch.setattr(factory, "build_vehicle_detector", build_vehicle)
    monkeypatch.setattr(factory, "build_plate_detector", build_plate)
    monkeypatch.setattr(factory, "build_ocr", lambda _: ocr)
    cfg = AppConfig()
    cfg.source.realtime = False
    with pytest.raises(KeyboardInterrupt):
        factory.build_parallel_pipeline(cfg, 1)
    assert len(detectors) == 4
    for detector in detectors:
        detector.close.assert_called_once()
    ocr.close.assert_called_once()


@pytest.mark.parametrize("failing", ["pipeline", "store", "source"])
def test_falha_na_limpeza_do_lote_marca_o_video_e_fecha_o_resto(monkeypatch, failing):
    import lprts.factory as factory
    from lprts.core.batch import process_one_video

    source = SimpleNamespace(close=Mock())
    store = SimpleNamespace(close=Mock(), session_id="test", session_summary=lambda: {})
    pipeline = SimpleNamespace(
        run=lambda _: iter(()), finalize=Mock(), stop=Mock(),
        metrics=SimpleNamespace(snapshot=lambda: SimpleNamespace(counts_by_class={})),
    )
    action = {"pipeline": pipeline.stop, "store": store.close, "source": source.close}[failing]
    action.side_effect = RuntimeError("teardown failed")
    monkeypatch.setattr(factory, "build_source", lambda _: source)
    monkeypatch.setattr(factory, "build_store", lambda _: store)
    monkeypatch.setattr(factory, "build_pipeline", lambda *_: pipeline)
    monkeypatch.setattr(factory, "open_session", Mock())
    monkeypatch.setattr(factory, "close_session", Mock())
    result = process_one_video("test.mp4", AppConfig().to_dict(), db_path="unused.db")
    assert not result.ok
    assert "teardown failed" in result.error
    pipeline.stop.assert_called_once()
    store.close.assert_called_once()
    source.close.assert_called_once()
