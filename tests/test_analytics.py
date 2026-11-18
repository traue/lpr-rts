"""Persistencia, metricas e analitica de trafego."""

from __future__ import annotations

import numpy as np
import pytest

from lprts.analytics.metrics import MetricsCollector
from lprts.analytics.store import ResultStore
from lprts.analytics.traffic import Calibration, TrafficAnalyzer, VirtualLine
from lprts.core.taxonomy import ObjectClass
from lprts.core.types import (
    Frame,
    FrameResult,
    PlateLayout,
    PlateRead,
    StageTiming,
    TrackState,
)


def trilha(track_id: int = 1, placa: str = "ABC1D23") -> TrackState:
    t = TrackState(
        track_id=track_id, bbox=(10, 10, 110, 60), score=0.9, object_class=ObjectClass.CARRO
    )
    t.plate_text = placa
    t.plate_confidence = 0.87
    t.plate_layout = PlateLayout.MERCOSUL
    t.centroids = [(10.0, 10.0), (20.0, 20.0)]
    t.timestamps = [0.0, 0.033]
    t.reads = [PlateRead(text=placa, confidence=0.87)]
    return t


class TestResultStore:
    def test_grava_e_le_eventos(self, tmp_db):
        with ResultStore(tmp_db) as store:
            store.start_session("v.mp4", 30.0, 1920, 1080, 100)
            store.add_event(trilha(1, "ABC1D23"), 10, 0.33)
            store.add_event(trilha(2, "XYZ9K88"), 20, 0.66)
            resumo = store.session_summary()
        assert resumo["n_events"] == 2
        assert resumo["n_unique_plates"] == 2

    def test_flush_sincroniza_antes_da_leitura(self, tmp_db):
        """Regressao: a fila vazia nao garante que o lote foi commitado."""
        with ResultStore(tmp_db, batch_size=1000) as store:
            store.start_session("v.mp4")
            for i in range(50):
                store.add_event(trilha(i, f"ABC{i % 10}D23"), i, i / 30)
            assert len(store.query("SELECT * FROM events")) == 50

    def test_upsert_de_trilha_e_idempotente(self, tmp_db):
        with ResultStore(tmp_db) as store:
            store.start_session("v.mp4")
            t = trilha()
            store.upsert_track(t)
            t.plate_confidence = 0.99
            store.upsert_track(t)
            linhas = store.query("SELECT plate_confidence FROM tracks")
        assert len(linhas) == 1
        assert linhas[0]["plate_confidence"] == pytest.approx(0.99)

    def test_view_de_desempenho(self, tmp_db):
        """A vazao e quadros/tempo; a latencia e a ponta a ponta."""
        with ResultStore(tmp_db) as store:
            store.start_session("v.mp4")
            for i in range(10):
                fr = FrameResult(
                    frame=Frame(i, np.zeros((4, 4, 3), np.uint8), 0.0, i / 30),
                    timing=StageTiming(detect_ms=12.0, ocr_ms=4.0, total_ms=20.0,
                                       latency_ms=35.0),
                )
                # Um quadro pronto a cada 50 ms -> 20 quadros/s.
                store.add_frame_telemetry(fr, 50.0, t0=0.0, done_t=0.05 * i)
            perf = store.query("SELECT * FROM v_session_performance")[0]
        assert perf["n_frames"] == 10
        assert perf["throughput_fps"] == pytest.approx(20.0)
        assert perf["avg_latency_ms"] == pytest.approx(35.0)
        assert perf["avg_service_ms"] == pytest.approx(20.0)

    def test_leituras_brutas_podem_ser_desligadas(self, tmp_db):
        with ResultStore(tmp_db, store_raw_reads=False) as store:
            store.start_session("v.mp4")
            store.add_read(1, PlateRead(text="ABC1D23", confidence=0.8), 0, 0.0)
            assert store.query("SELECT * FROM reads") == []


class TestTrafficAnalyzer:
    def test_conta_cruzamento_uma_unica_vez(self):
        ta = TrafficAnalyzer(lines=[VirtualLine("l1", (0, 100), (200, 100))])
        t = TrackState(track_id=1, bbox=(90, 50, 110, 70), score=0.9)
        eventos = []
        for i in range(20):
            y = 50 + i * 8
            t.bbox = (90, y - 10, 110, y + 10)
            t.centroids.append((100.0, float(y)))
            t.timestamps.append(i / 30)
            eventos += ta.update([t], i, i / 30)
        assert len(eventos) == 1
        assert ta.lines[0].total == 1

    def test_sem_calibracao_nao_inventa_velocidade(self):
        ta = TrafficAnalyzer(calibration=Calibration())
        t = TrackState(track_id=1, bbox=(0, 0, 10, 10), score=0.9)
        t.centroids = [(float(i * 10), 0.0) for i in range(10)]
        t.timestamps = [i / 30 for i in range(10)]
        ta.update([t], 0, 0.0)
        assert t.speed_kmh is None

    def test_com_homografia_estima_velocidade(self):
        cal = Calibration(
            image_points=[(400, 700), (880, 700), (760, 400), (520, 400)],
            world_points=[(0, 0), (3.5, 0), (3.5, 20.0), (0, 20.0)],
        )
        ta = TrafficAnalyzer(calibration=cal)
        t = TrackState(track_id=1, bbox=(600, 400, 700, 480), score=0.9)
        for i in range(20):
            y = 420 + i * 10
            t.centroids.append((650.0, float(y)))
            t.timestamps.append(i / 30.0)
        ta.update([t], 0, 0.0)
        assert t.speed_kmh is not None and t.speed_kmh > 0

    def test_sentido_do_movimento(self):
        """Direcoes sao NA IMAGEM (direita/esquerda/baixo/cima), nao pontos cardeais."""
        ta = TrafficAnalyzer()
        t = TrackState(track_id=1, bbox=(0, 0, 10, 10), score=0.9)
        t.centroids = [(float(i * 10), 0.0) for i in range(10)]
        t.timestamps = [i / 30 for i in range(10)]
        ta.update([t], 0, 0.0)
        assert t.direction == "direita"

    def test_objeto_parado(self):
        ta = TrafficAnalyzer()
        t = TrackState(track_id=1, bbox=(0, 0, 10, 10), score=0.9)
        t.centroids = [(100.0, 100.0)] * 10
        t.timestamps = [i / 30 for i in range(10)]
        ta.update([t], 0, 0.0)
        assert t.direction == "parado"

    def test_esquece_trilhas_encerradas(self):
        ta = TrafficAnalyzer(lines=[VirtualLine("l1", (0, 100), (200, 100))])
        t = TrackState(track_id=7, bbox=(90, 50, 110, 70), score=0.9)
        ta.update([t], 0, 0.0)
        ta.forget({7})
        assert 7 not in ta._prev_centroid


class TestMetrics:
    def test_percentis_e_taxas(self):
        m = MetricsCollector()
        for i in range(100):
            m.on_frame(total_ms=10.0 + i * 0.1, detect_ms=8.0, ocr_ms=2.0, n_active_tracks=2)
        for i in range(20):
            m.on_read(valid_syntax=i % 5 != 0)
        m.on_event("ABC1D23")
        m.on_event("ABC1D23")
        s = m.snapshot()
        assert s.frames_processed == 100
        assert s.latency.p50 < s.latency.p95 < s.latency.max
        assert s.syntax_valid_rate == pytest.approx(0.8)
        assert s.n_unique_plates == 1 and s.n_events == 2

    def test_reset(self):
        m = MetricsCollector()
        m.on_frame(total_ms=10.0)
        m.reset()
        assert m.snapshot().frames_processed == 0
