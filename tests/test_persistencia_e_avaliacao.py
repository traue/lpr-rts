"""Persistencia e avaliacao: limpeza segura, migracao, gravacao critica e elegibilidade."""

import sqlite3
import threading
import time
from contextlib import suppress

import numpy as np
import pytest

from lprts.analytics import store as store_module
from lprts.analytics.metrics import MetricsCollector
from lprts.analytics.store import ResultStore, clear_results
from lprts.core.types import Frame, FrameResult, PlateLayout, PlateRead, TrackState
from lprts.eval.groundtruth import GroundTruth, GTAnnotation
from lprts.eval.session import evaluate_session, tracking_metrics
from lprts.optimize.refusion import TrackReads, attach_ground_truth, evaluate_fusion


@pytest.mark.parametrize("columns", ["token TEXT, account_id INTEGER", "session_id TEXT"])
def test_limpeza_recusa_tabela_sessions_de_outro_programa(tmp_path, columns):
    db = tmp_path / "foreign.db"
    with sqlite3.connect(db) as con:
        con.execute(f"CREATE TABLE sessions ({columns})")
        con.execute("CREATE TABLE invoices (amount REAL)")
        con.execute("INSERT INTO invoices VALUES (123.45)")
    with pytest.raises(ValueError, match="nao e um banco do LPR-RTS"):
        clear_results(db)
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT amount FROM invoices").fetchone() == (123.45,)


@pytest.mark.parametrize("spoof_session", [False, True])
def test_abrir_nao_altera_banco_de_outro_programa(tmp_path, spoof_session):
    db = tmp_path / "foreign-open.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE invoices (amount REAL)")
        if spoof_session:
            con.execute("CREATE TABLE sessions (session_id TEXT)")
            con.execute("CREATE TABLE events (id INTEGER)")
            con.execute("PRAGMA user_version=2")
        original = con.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    with pytest.raises(RuntimeError, match="esquema"):
        ResultStore(db).open()
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall() == original


def test_migracao_que_falha_volta_atras_e_reabre(tmp_path, monkeypatch):
    db = tmp_path / "migration.db"
    with sqlite3.connect(db) as con:
        con.executescript(store_module._load_schema())
        for sql in (
            "DROP VIEW v_unique_plates", "DROP VIEW v_decisions",
            "DROP INDEX idx_events_track", "ALTER TABLE events DROP COLUMN decision",
            "ALTER TABLE events DROP COLUMN revision",
            "ALTER TABLE tracks DROP COLUMN first_published_text",
            "ALTER TABLE tracks DROP COLUMN n_revisions",
            "ALTER TABLE tracks DROP COLUMN published_decision",
        ):
            con.execute(sql)
        if "decision_elapsed_ms" in {r[1] for r in con.execute("PRAGMA table_info(events)")}:
            con.execute("ALTER TABLE events DROP COLUMN decision_elapsed_ms")
        con.execute("PRAGMA user_version=2")
    connect = sqlite3.connect

    class MigracaoQueFalha(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == store_module._MIGRACOES[2][1]:
                raise sqlite3.OperationalError("injected migration failure")
            return super().execute(sql, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store_module.sqlite3, "connect",
                      lambda *a, **k: connect(*a, **k, factory=MigracaoQueFalha))
        store = ResultStore(db)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            store.open()
        assert store._conn is None
    with connect(db) as con:
        assert "decision" not in {r[1] for r in con.execute("PRAGMA table_info(events)")}
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
    with ResultStore(db, async_writes=False):
        pass


def _track():
    return TrackState(track_id=1, bbox=(0, 0, 100, 20), score=.9, plate_text="ABC1D23")


def test_fila_critica_cheia_falha_logo_e_fecha_com_seguranca(tmp_path, monkeypatch):
    store = ResultStore(tmp_path / "pressure.db", queue_size=1, batch_size=1)
    entered, release = threading.Event(), threading.Event()
    original = store._flush

    def blocked(batch):
        if batch:
            entered.set()
            assert release.wait(5)
        original(batch)

    monkeypatch.setattr(store, "_flush", blocked)
    store.open()
    store.start_session("test")
    try:
        store.add_event(_track(), 0, 0)
        assert entered.wait(1)
        store.add_event(_track(), 1, .1)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="[Ff]ila|queue|saturad"):
            store.add_event(_track(), 2, .2)
        assert time.monotonic() - start < .5
        with pytest.raises(RuntimeError):
            store.close(timeout=.01)
        assert store._conn is not None
        assert store._thread is not None and store._thread.is_alive()
    finally:
        release.set()
        with suppress(RuntimeError):
            store.close()
    assert store._conn is None


@pytest.mark.parametrize("async_writes", [True, False])
def test_erro_sql_em_resultado_critico_chega_ao_produtor(tmp_path, async_writes):
    store = ResultStore(tmp_path / "bad-write.db", async_writes=async_writes, batch_size=1).open()
    try:
        with pytest.raises(RuntimeError):
            store._enqueue("events", "INSERT INTO missing_table VALUES (?)", (1,))
            store.flush(timeout=1)
    finally:
        with suppress(RuntimeError):
            store.close()
    assert store._conn is None


def test_revisao_tira_da_contagem_so_a_placa_substituida():
    metrics = MetricsCollector()
    metrics.on_event("ABC1D23")
    metrics.on_event("ABC1D23")
    metrics.on_event("XYZ9K88", replaces="ABC1D23")
    assert metrics.snapshot().n_unique_plates == 2
    metrics.on_event("XYZ9K88", replaces="ABC1D23")
    assert metrics.snapshot().n_unique_plates == 1
    assert metrics.snapshot().n_events == 2


def test_ajuste_ignora_trilhas_dificeis_e_conta_falsos_positivos():
    annotations = [
        GTAnnotation(0, (0, 0, 100, 20), "ABC1D23", track_id=1),
        GTAnnotation(0, (200, 0, 300, 20), "XYZ9K88", track_id=2, difficult=True),
    ]
    tracks = [TrackReads(a.track_id, "s", [PlateRead(
        text=a.plate_text, confidence=.9, valid_syntax=True, layout=PlateLayout.MERCOSUL,
        bbox=a.bbox, frame_index=0,
    )]) for a in annotations]
    tracks.append(TrackReads(3, "s", [PlateRead(
        text="ZZZ9Z99", confidence=.9, valid_syntax=True, layout=PlateLayout.MERCOSUL,
        bbox=(400, 0, 500, 20), frame_index=0,
    )]))
    attach_ground_truth(tracks, GroundTruth(annotations=annotations))
    metrics = evaluate_fusion(tracks, n_vehicles=1, min_reads=1)
    assert metrics.recall == 1
    assert metrics.n_correct == 1 and metrics.n_wrong == 1
    assert metrics.precision == .5 and metrics.f1 <= 1


def test_zero_deteccoes_sao_erros_medidos(tmp_path):
    db = tmp_path / "missed.db"
    with ResultStore(db, async_writes=False) as store:
        store.start_session("test", config={"analytics": {"store_detections": True}})
        store.add_frame_telemetry(FrameResult(
            frame=Frame(0, np.zeros((20, 20, 3), dtype=np.uint8), 0), detection_ran=True,
        ), 30)
        sid = store.session_id
    gt = GroundTruth(annotations=[
        GTAnnotation(0, (0, 0, 100, 20), "ABC1D23", track_id=1),
        GTAnnotation(0, (0, 0, 100, 100), "", track_id=1, kind="object"),
    ])
    report = evaluate_session(db, sid, gt, n_boot=10)
    assert report.detection is not None and report.detection.fn == 1
    assert report.detection.recall == 0
    assert report.object_detection is not None and report.object_detection.fn == 1
    assert not any("store_detections desligado" in w for w in report.warnings)


def test_mota_filtra_o_limiar_antes_de_atribuir():
    box0 = (0, 0, 100, 100)
    box1 = (43, 0, 143, 100)
    box2 = (-43, 0, 57, 100)
    result = tracking_metrics(
        {1: {0: box0}, 2: {0: box2}},
        {"a": {0: box0}, "b": {0: box1}}, {0}, iou_threshold=.2,
    )
    assert result["tp"] == 2 and result["fp"] == result["fn"] == 0


def test_migracao_do_esquema_3_deixa_o_relogio_desconhecido(tmp_path):
    db = tmp_path / "v3.db"
    with ResultStore(db, async_writes=False) as store:
        store.start_session("test")
        store.add_event(_track(), 0, 0)
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE events DROP COLUMN decision_elapsed_ms")
        con.execute("PRAGMA user_version=3")
    with ResultStore(db, async_writes=False) as store:
        assert store.query("SELECT decision_elapsed_ms FROM events")[0][0] is None


def test_falha_da_escritora_marca_a_sessao_e_o_flush_nao_esconde(tmp_path, monkeypatch):
    db = tmp_path / "writer.db"
    store = ResultStore(db, batch_size=1).open()
    store.start_session("test")

    def fail(batch):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_flush", fail)
    try:
        with pytest.raises(RuntimeError, match="disk unavailable"):
            store.add_event(_track(), 0, 0)
            store.flush(timeout=1)
    finally:
        with pytest.raises(RuntimeError, match="disk unavailable"):
            store.close(timeout=1)
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT end_reason FROM sessions").fetchone()[0] == "erro_persistencia"
