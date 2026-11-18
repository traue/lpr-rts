"""Processamento em lote: consolidacao dos bancos e backends."""

from __future__ import annotations

import os
import sqlite3

import cv2
import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.config import AppConfig
from lprts.core.batch import (
    BACKENDS,
    BatchResult,
    VideoResult,
    merge_databases,
    run_batch,
)
from lprts.core.taxonomy import ObjectClass
from lprts.core.types import PlateLayout, TrackState


def _trilha(track_id: int, placa: str) -> TrackState:
    t = TrackState(
        track_id=track_id, bbox=(10, 10, 110, 60), score=0.9, object_class=ObjectClass.CARRO
    )
    t.plate_text = placa
    t.plate_confidence = 0.9
    t.plate_layout = PlateLayout.MERCOSUL
    t.centroids = [(1.0, 2.0)]
    t.timestamps = [0.0]
    return t


def _shard(path, n_eventos: int, sessao: str) -> None:
    """Cria um fragmento com ids comecando em 1 - a condicao que expunha o bug."""
    with ResultStore(path, session_id=sessao, async_writes=False) as store:
        store.start_session(f"{sessao}.mp4")
        for i in range(n_eventos):
            t = _trilha(i, f"ABC{i % 10}D23")
            store.add_event(t, i, i / 30.0)
            store.upsert_track(t)


class TestMergeDatabases:
    def test_nao_perde_linhas_com_ids_colidentes(self, tmp_path):
        """Regressao do bug mais serio deste modulo.

        Cada fragmento numera suas linhas a partir de 1. Um
        ``INSERT OR IGNORE ... SELECT *`` levaria os ids junto, as chaves
        colidiriam e as linhas seriam descartadas **em silencio** - o banco
        consolidado sairia com uma fracao dos dados e nada indicaria o problema.
        """
        shards = []
        for i in range(3):
            p = tmp_path / f"shard_{i}.db"
            _shard(p, n_eventos=5, sessao=f"s{i}")
            shards.append(p)

        alvo = merge_databases(shards, tmp_path / "consolidado.db")
        con = sqlite3.connect(str(alvo))
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 15
        assert con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 15
        assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
        con.close()

    def test_ids_sao_reatribuidos_e_unicos(self, tmp_path):
        shards = [tmp_path / f"s{i}.db" for i in range(2)]
        for i, p in enumerate(shards):
            _shard(p, 4, f"sess{i}")
        alvo = merge_databases(shards, tmp_path / "m.db")
        con = sqlite3.connect(str(alvo))
        ids = [r[0] for r in con.execute("SELECT id FROM events")]
        con.close()
        assert len(ids) == len(set(ids)) == 8

    def test_fragmento_ausente_e_ignorado(self, tmp_path):
        p = tmp_path / "existe.db"
        _shard(p, 3, "s")
        alvo = merge_databases([p, tmp_path / "nao_existe.db"], tmp_path / "m.db")
        con = sqlite3.connect(str(alvo))
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
        con.close()

    def test_esquema_completo_no_alvo(self, tmp_path):
        p = tmp_path / "s.db"
        _shard(p, 1, "s")
        alvo = merge_databases([p], tmp_path / "m.db")
        con = sqlite3.connect(str(alvo))
        views = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        con.close()
        assert {"v_unique_plates", "v_object_counts", "v_category_share"} <= views


class TestBatchResult:
    def test_agregacao(self):
        r = BatchResult(
            results=[
                VideoResult("a.mp4", ok=True, n_frames=100, counts_by_class={"carro": 3}),
                VideoResult("b.mp4", ok=True, n_frames=200, counts_by_class={"carro": 2, "pessoa": 5}),
                VideoResult("c.mp4", ok=False, error="boom"),
            ],
            elapsed_s=10.0,
        )
        assert len(r.ok) == 2 and len(r.failed) == 1
        assert r.total_frames == 300
        assert r.aggregate_fps == pytest.approx(30.0)
        assert r.counts_by_class == {"pessoa": 5, "carro": 5}

    def test_resumo_menciona_falhas(self):
        r = BatchResult(results=[VideoResult("x.mp4", ok=False, error="codec")], elapsed_s=1.0)
        assert "FALHA" in r.summary() and "codec" in r.summary()


@pytest.fixture
def videos(tmp_path, plate_image):
    caminhos = []
    for v in range(2):
        p = tmp_path / f"v{v}.mp4"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
        for i in range(40):
            cena = np.full((360, 640, 3), 95, dtype=np.uint8)
            largura = 120 + i * 3
            altura = int(largura * 0.33)
            cena[150 : 150 + altura, 260 : 260 + largura] = cv2.resize(
                plate_image(f"ABC{v}D23"), (largura, altura)
            )
            w.write(cena)
        w.release()
        caminhos.append(str(p))
    return caminhos


class TestRunBatch:
    def _config(self) -> AppConfig:
        cfg = AppConfig()
        cfg.detector.detect_vehicles = False
        cfg.detector.plate_detector = "classic"
        cfg.ocr.engine = "template"
        cfg.ocr.min_plate_width = 55
        cfg.analytics.save_crops = False
        return cfg

    def test_sequencial(self, tmp_path, videos):
        r = run_batch(videos, self._config(), tmp_path / "out", backend="sequential")
        assert len(r.ok) == 2
        assert r.total_frames == 80
        assert r.merged_db

    def test_backend_invalido(self, tmp_path, videos):
        with pytest.raises(ValueError):
            run_batch(videos, self._config(), tmp_path / "o", backend="inexistente")

    def test_lista_vazia(self, tmp_path):
        with pytest.raises(ValueError):
            run_batch([], self._config(), tmp_path / "o")

    def test_video_invalido_nao_aborta_o_lote(self, tmp_path, videos):
        """Um arquivo corrompido no meio do corpus nao pode derrubar tudo."""
        r = run_batch(
            [*videos, str(tmp_path / "inexistente.mp4")],
            self._config(), tmp_path / "out", backend="sequential",
        )
        assert len(r.ok) == 2 and len(r.failed) == 1

    def test_backends_declarados(self):
        assert set(BACKENDS) == {"sequential", "process", "mpi"}


# ---------------------------------------------------------------------------
# Reparticao de threads e aviso de paralelismo contraproducente
# ---------------------------------------------------------------------------

def test_limit_native_threads_serial_nao_mexe_em_nada(monkeypatch):
    """Com um worker nao ha disputa: devolve a maquina inteira e nao toca no ambiente."""
    from lprts.runtime import _VARS_THREADS, limit_native_threads

    monkeypatch.setattr("os.cpu_count", lambda: 8)
    for var in _VARS_THREADS:
        monkeypatch.delenv(var, raising=False)

    assert limit_native_threads(1) == 8
    assert all(var not in os.environ for var in _VARS_THREADS)


def test_limit_native_threads_reparte_os_nucleos(monkeypatch):
    from lprts.runtime import _VARS_THREADS, limit_native_threads

    monkeypatch.setattr("os.cpu_count", lambda: 8)
    for var in _VARS_THREADS:
        monkeypatch.delenv(var, raising=False)

    assert limit_native_threads(4) == 2
    assert all(os.environ[var] == "2" for var in _VARS_THREADS)


def test_limit_native_threads_nunca_devolve_zero(monkeypatch):
    """Mais workers que nucleos ainda precisa de pelo menos uma thread cada."""
    from lprts.runtime import limit_native_threads

    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert limit_native_threads(16) == 1


def test_aviso_de_paralelismo_contraproducente(monkeypatch, caplog):
    """Detector em PyTorch + poucos nucleos: o usuario precisa ser avisado."""
    from lprts.core.batch import _avisar_se_contraproducente

    monkeypatch.setattr("os.cpu_count", lambda: 4)
    cfg = AppConfig()
    cfg.detector.detect_vehicles = True

    with caplog.at_level("WARNING"):
        _avisar_se_contraproducente(cfg, n_workers=3)
    assert "MAIS LENTOS" in caplog.text


def test_sem_aviso_quando_o_worker_e_leve(monkeypatch, caplog):
    from lprts.core.batch import _avisar_se_contraproducente

    monkeypatch.setattr("os.cpu_count", lambda: 4)
    cfg = AppConfig()
    cfg.detector.detect_vehicles = False
    cfg.detector.plate_detector = "classic"

    with caplog.at_level("WARNING"):
        _avisar_se_contraproducente(cfg, n_workers=3)
    assert caplog.text == ""


def test_sem_aviso_quando_ha_nucleos_de_sobra(monkeypatch, caplog):
    from lprts.core.batch import _avisar_se_contraproducente

    monkeypatch.setattr("os.cpu_count", lambda: 32)
    cfg = AppConfig()
    cfg.detector.detect_vehicles = True

    with caplog.at_level("WARNING"):
        _avisar_se_contraproducente(cfg, n_workers=3)
    assert caplog.text == ""
