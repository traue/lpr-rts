"""Testes de integracao: pipeline de ponta a ponta e captura de video."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.capture.ringbuffer import DropOldestQueue
from lprts.capture.source import VideoSource
from lprts.config import AppConfig
from lprts.core.pipeline import LPRPipeline
from lprts.detect.classic_plate import ClassicPlateDetector
from lprts.ocr.engines import build_ocr_engine


@pytest.fixture
def video_sintetico(tmp_path, plate_image):
    """Video curto com uma placa que se aproxima da camera."""
    caminho = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    assert writer.isOpened()
    texto = "ABC1D23"
    for i in range(60):
        cena = np.full((360, 640, 3), 95, dtype=np.uint8)
        largura = 120 + i * 3           # a placa cresce a cada quadro
        altura = int(largura * 0.33)
        x1 = 260 - largura // 2
        y1 = 160
        cena[y1 : y1 + altura, x1 : x1 + largura] = cv2.resize(
            plate_image(texto), (largura, altura)
        )
        writer.write(cena)
    writer.release()
    return caminho, texto


class TestDropOldestQueue:
    def test_descarta_o_mais_antigo(self):
        q = DropOldestQueue(maxsize=2)
        for i in range(5):
            q.put(i)
        assert q.dropped == 3
        assert q.get() == 3          # os antigos sairam

    def test_get_latest_pula_a_fila(self):
        """No modo tempo real, a interface quer o quadro atual, nao a fila."""
        q = DropOldestQueue(maxsize=10)
        for i in range(5):
            q.put(i)
        assert q.get_latest() == 4
        assert len(q) == 0

    def test_fechada_devolve_none(self):
        q = DropOldestQueue(maxsize=2)
        q.close()
        assert q.get(timeout=0.05) is None

    def test_maxsize_invalido(self):
        with pytest.raises(ValueError):
            DropOldestQueue(maxsize=0)


class TestVideoSource:
    def test_le_todos_os_quadros_no_modo_offline(self, video_sintetico):
        caminho, _ = video_sintetico
        with VideoSource(caminho) as src:
            assert src.fps == pytest.approx(30.0, abs=0.1)
            assert src.width == 640 and src.height == 360
            assert sum(1 for _ in src.frames()) == 60

    def test_indices_e_tempos_crescem(self, video_sintetico):
        caminho, _ = video_sintetico
        with VideoSource(caminho) as src:
            quadros = list(src.frames())
        assert [q.index for q in quadros[:5]] == [0, 1, 2, 3, 4]
        assert quadros[10].source_pts > quadros[0].source_pts

    def test_redimensiona(self, video_sintetico):
        caminho, _ = video_sintetico
        with VideoSource(caminho, resize_width=320) as src:
            assert next(iter(src.frames())).image.shape[1] == 320

    def test_arquivo_inexistente(self):
        with pytest.raises(FileNotFoundError):
            VideoSource("/caminho/que/nao/existe.mp4").open()


class TestPipelineIntegracao:
    def _config(self, tmp_path) -> AppConfig:
        cfg = AppConfig()
        cfg.detector.detect_vehicles = False       # sem pesos YOLO nos testes
        cfg.ocr.min_plate_width = 55
        cfg.ocr.ocr_every_n = 1
        cfg.ocr.max_ocr_per_frame = 4
        cfg.analytics.db_path = str(tmp_path / "t.db")
        cfg.analytics.crops_dir = str(tmp_path / "crops")
        cfg.analytics.save_crops = False
        return cfg

    def test_le_a_placa_de_ponta_a_ponta(self, tmp_path, video_sintetico):
        caminho, esperado = video_sintetico
        cfg = self._config(tmp_path)
        cfg.source.uri = str(caminho)

        with VideoSource(caminho) as src, ResultStore(cfg.analytics.db_path) as store:
            store.start_session(str(caminho), src.fps, src.width, src.height)
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12),
                build_ocr_engine("template"), store,
            ).start()
            resultados = list(pipe.run(src))
            pipe.finalize()
            placas = {
                r["plate_text"]
                for r in store.query("SELECT DISTINCT plate_text FROM events WHERE plate_text<>''")
            }
            pipe.stop()

        assert len(resultados) == 60
        assert esperado in placas

    def test_telemetria_e_gravada(self, tmp_path, video_sintetico):
        caminho, _ = video_sintetico
        cfg = self._config(tmp_path)
        with VideoSource(caminho) as src, ResultStore(cfg.analytics.db_path) as store:
            store.start_session(str(caminho))
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12),
                build_ocr_engine("template"), store,
            ).start()
            for _ in pipe.run(src):
                pass
            pipe.finalize()
            perf = store.query("SELECT * FROM v_session_performance")
            pipe.stop()
        assert perf and perf[0]["n_frames"] == 60
        assert perf[0]["avg_latency_ms"] > 0

    def test_metricas_coerentes(self, tmp_path, video_sintetico):
        caminho, _ = video_sintetico
        cfg = self._config(tmp_path)
        with VideoSource(caminho) as src:
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12),
                build_ocr_engine("template"), None,
            ).start()
            for _ in pipe.run(src):
                pass
            pipe.finalize()
            m = pipe.metrics.snapshot()
            pipe.stop()
        assert m.frames_processed == 60
        assert m.n_reads > 0
        assert 0.0 <= m.syntax_valid_rate <= 1.0

    def test_modo_so_placa_rastreia_placas(self, tmp_path, video_sintetico):
        """Sem detector de veiculos, as placas devem virar trilhas proprias.

        Regressao: antes o quadro inteiro era tratado como um unico objeto, e
        todas as placas da cena votavam juntas numa trilha so.
        """
        caminho, _ = video_sintetico
        cfg = self._config(tmp_path)
        with VideoSource(caminho) as src:
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12),
                build_ocr_engine("template"), None,
            ).start()
            assert pipe._plate_only_mode
            trilhas_por_quadro = [len(r.tracks) for r in pipe.run(src)]
            pipe.finalize()
            pipe.stop()
        # Uma placa em cena -> no maximo uma trilha ativa por quadro.
        assert max(trilhas_por_quadro) <= 2

    def test_limite_de_quadros_respeitado(self, tmp_path, video_sintetico):
        caminho, _ = video_sintetico
        cfg = self._config(tmp_path)
        cfg.source.max_frames = 10
        with VideoSource(caminho) as src:
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12), None, None
            ).start()
            assert len(list(pipe.run(src))) == 10
            pipe.stop()

    def test_sem_ocr_nao_quebra(self, tmp_path, video_sintetico):
        """O pipeline deve rodar como detector puro, sem motor de OCR."""
        caminho, _ = video_sintetico
        cfg = self._config(tmp_path)
        with VideoSource(caminho) as src:
            pipe = LPRPipeline(
                cfg, None, ClassicPlateDetector(min_height=12), None, None
            ).start()
            resultados = list(pipe.run(src))
            pipe.finalize()
            pipe.stop()
        assert len(resultados) == 60
        assert all(r.reads == [] for r in resultados)


class TestBuscaDePlacaPorVeiculo:
    """A placa so e procurada em veiculos largos o bastante para ser legivel."""

    def _selecionar(self, razao: float, teto: int = 0):
        from lprts.core.types import Detection

        cfg = AppConfig()
        cfg.ocr.min_plate_width = 60
        cfg.detector.plate_width_ratio = razao
        cfg.detector.max_rois_per_frame = teto
        pipe = LPRPipeline(cfg, None, ClassicPlateDetector(), build_ocr_engine("template"), None)
        objetos = [
            Detection(bbox=(0, 0, 300, 200), score=0.9, label="car", class_id=2),
            Detection(bbox=(0, 0, 119, 80), score=0.9, label="car", class_id=2),  # distante
            Detection(bbox=(0, 0, 130, 90), score=0.9, label="motorcycle", class_id=3),
            Detection(bbox=(0, 0, 400, 300), score=0.9, label="person", class_id=0),
        ]
        rois, puladas = pipe._select_rois(objetos)
        return [r.bbox[2] for r in rois], puladas

    def test_veiculo_estreito_demais_nao_tem_placa_procurada(self):
        # 60 px de placa / 0,5 = 120 px de veiculo; pessoa nunca porta placa.
        assert self._selecionar(0.5) == ([300, 130], 1)

    def test_razao_zero_procura_em_todos(self):
        assert self._selecionar(0.0) == ([300, 119, 130], 0)

    def test_teto_conta_junto_com_os_estreitos(self):
        assert self._selecionar(0.5, teto=1) == ([300], 2)


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation so existe no macOS")
class TestDecodificadorDeHardware:
    """Arquivos pelo AVFoundation (VideoToolbox): o FFmpeg do OpenCV decodifica em
    software e um quadro 4K HEVC custava 46-48 ms - mais que o prazo de 33 ms."""

    SINTETICO = Path(__file__).resolve().parent.parent / "data" / "videos" / "sintetico.mp4"

    def test_auto_usa_avfoundation_e_entrega_os_mesmos_quadros(self):
        if not self.SINTETICO.exists():
            pytest.skip("video sintetico ausente")
        with VideoSource(self.SINTETICO) as hw, VideoSource(self.SINTETICO, backend="ffmpeg") as sw:
            assert hw.backend_name == "AVFOUNDATION" and sw.backend_name == "FFMPEG"
            a, b = list(hw.frames()), list(sw.frames())
        assert len(a) == len(b) == 400
        assert [f.source_pts for f in a] == pytest.approx([f.source_pts for f in b])
        assert np.abs(a[200].image.astype(int) - b[200].image.astype(int)).mean() < 4

    def test_busca_reabre_pelo_ffmpeg(self):
        if not self.SINTETICO.exists():
            pytest.skip("video sintetico ausente")
        with VideoSource(self.SINTETICO) as src:
            assert src.seek(100)
            assert src.backend_name == "FFMPEG"
            assert next(iter(src.frames())).source_pts == pytest.approx(100 / 30, abs=0.02)

    def test_backend_desconhecido_e_erro(self):
        with pytest.raises(ValueError, match="Backend de video"):
            VideoSource("x.mp4", backend="gstreamer")
