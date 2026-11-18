"""Integracao: partida segura, vida das threads e comandos instalados."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lprts.analytics.store import ResultStore
from lprts.config import AppConfig


def test_modelos_que_falham_preservam_o_historico_e_fecham_a_fonte(tmp_path, monkeypatch):
    import lprts.factory as factory
    from lprts.cli import _build_runtime

    db = tmp_path / "previous.db"
    with ResultStore(db, async_writes=False) as store:
        store.start_session("previous.mp4")
        store.finish_session()
    cfg = AppConfig()
    cfg.analytics.db_path = str(db)
    cfg.detector.detect_vehicles = False
    cfg.detector.plate_detector = "classic"
    cfg.ocr.engine = "invalid-engine"
    source = SimpleNamespace(close=Mock())
    monkeypatch.setattr(factory, "build_source", lambda _: source)
    monkeypatch.setattr(factory, "check_calibration", lambda *_: None)

    with pytest.raises(ValueError):
        _build_runtime(cfg, clear=True)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    source.close.assert_called_once()


def test_montagem_parcial_fecha_os_componentes_criados(monkeypatch):
    import lprts.factory as factory

    vehicle, plate = Mock(), Mock()
    monkeypatch.setattr(factory, "build_vehicle_detector", lambda _: vehicle)
    monkeypatch.setattr(factory, "build_plate_detector", lambda _: plate)
    monkeypatch.setattr(factory, "build_ocr", Mock(side_effect=ValueError("bad OCR")))
    with pytest.raises(ValueError, match="bad OCR"):
        factory.build_pipeline(AppConfig())
    vehicle.close.assert_called_once()
    plate.close.assert_called_once()


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_janela_recusa_fechar_com_a_thread_viva(qapp):
    from PySide6.QtGui import QCloseEvent

    from lprts.ui.app import MainWindow

    window = SimpleNamespace(worker=SimpleNamespace(stop=Mock(), wait=lambda _: False))
    window.stop = lambda: MainWindow.stop(window)
    event = QCloseEvent()
    MainWindow.closeEvent(window, event)
    assert not event.isAccepted()


def test_cancelar_na_partida_preserva_os_resultados_anteriores(qapp, monkeypatch):
    import lprts.ui.worker as module

    worker = module.PipelineWorker(AppConfig())
    source = SimpleNamespace(close=Mock(), cancel=Mock())
    pipeline = SimpleNamespace(stop=Mock(), request_stop=Mock(), finalize=Mock())
    clear = Mock()
    monkeypatch.setattr(module, "build_source", lambda _: source)
    monkeypatch.setattr(module, "check_calibration", lambda *_: None)

    def build(*_):
        worker.stop()
        return pipeline

    monkeypatch.setattr(module, "build_configured_pipeline", build)
    monkeypatch.setattr(module, "clear_previous_results", clear)
    worker.run()
    clear.assert_not_called()
    pipeline.stop.assert_called_once()
    source.close.assert_called_once()
    assert not worker.running


def test_encerramento_fecha_tudo_apos_falha_ao_finalizar(qapp, monkeypatch):
    import lprts.ui.worker as module

    worker = module.PipelineWorker(AppConfig())
    worker.pipeline = SimpleNamespace(finalize=Mock(side_effect=RuntimeError("write failed")),
                                     stop=Mock())
    worker.store = SimpleNamespace(close=Mock())
    worker._source = SimpleNamespace(close=Mock())
    monkeypatch.setattr(module, "close_session", Mock())
    worker._teardown()
    worker.pipeline.stop.assert_called_once()
    worker.store.close.assert_called_once()
    worker._source.close.assert_called_once()


def test_modo_sem_janela_fecha_tudo_apos_falha_de_gravacao(monkeypatch):
    import lprts.cli as cli
    import lprts.factory as factory

    source, store, pipeline = Mock(), Mock(), Mock()
    pipeline.run.side_effect = RuntimeError("storage writer failed")
    pipeline.finalize.side_effect = RuntimeError("storage writer failed")
    monkeypatch.setattr(cli, "_build_runtime", lambda *_args, **_kwargs: (source, store, pipeline))
    monkeypatch.setattr(factory, "describe_pipeline", lambda *_: {})
    monkeypatch.setattr(factory, "open_session", Mock())
    monkeypatch.setattr(factory, "close_session", Mock())
    with pytest.raises(RuntimeError, match="storage writer failed"):
        cli._run_headless(AppConfig())
    pipeline.stop.assert_called_once()
    store.close.assert_called_once()
    source.close.assert_called_once()


def test_avaliacao_que_falha_marca_a_sessao(tmp_path, monkeypatch):
    import lprts.cli as cli
    import lprts.factory as factory

    cfg = AppConfig()
    cfg.analytics.db_path = str(tmp_path / "eval.db")
    store = ResultStore(cfg.analytics.db_path, async_writes=False).open()
    source = SimpleNamespace(close=Mock())
    pipeline = SimpleNamespace(run=Mock(side_effect=RuntimeError("detector failed")),
                               finalize=Mock(), stop=Mock(), stats=lambda: {})
    gt = tmp_path / "gt.json"
    gt.write_text('{"annotations": []}')
    monkeypatch.setattr(cli, "_build_runtime", lambda *_args, **_kwargs: (source, store, pipeline))
    monkeypatch.setattr(factory, "open_session", lambda *_args, **_kwargs: store.start_session("v.mp4"))
    with pytest.raises(RuntimeError, match="detector failed"):
        cli._evaluate(cfg, gt, tmp_path / "report.json")
    with sqlite3.connect(cfg.analytics.db_path) as conn:
        finished, reason = conn.execute("SELECT finished_at, end_reason FROM sessions").fetchone()
    assert finished is not None
    assert reason == "erro"
    pipeline.stop.assert_called_once()
    source.close.assert_called_once()


def test_gerador_e_benchmark_rodam_fora_do_checkout(tmp_path):
    # No CI o mesmo teste roda contra o wheel instalado, sem PYTHONPATH. Aqui,
    # garante que as importacoes nao dependem da pasta atual.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    video = tmp_path / "sample.mp4"
    generator = subprocess.run(
        [sys.executable, "-m", "lprts.synthetic", "--out", str(video), "--frames", "14",
         "--vehicles", "1", "--width", "320", "--height", "240"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert generator.returncode == 0, generator.stderr
    bench = subprocess.run(
        [sys.executable, "-m", "lprts.cli", "bench", "--source", str(video),
         "--frames", "2", "--no-vehicle-detector", "--plate-detector", "classic",
         "--ocr", "template"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert bench.returncode == 0, bench.stderr
    assert "quadros medidos" in bench.stdout


@pytest.mark.parametrize("module", ["lprts.analytics.store", "lprts.capture.source", "lprts.eval.report"])
def test_importar_nao_exige_datetime_utc_do_python_311(module, tmp_path):
    # O Python 3.10, minimo declarado, tem timezone.utc mas nao datetime.UTC.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    code = ("import datetime, importlib; "
            "datetime.__dict__.pop('UTC', None); "
            f"importlib.import_module({module!r})")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
