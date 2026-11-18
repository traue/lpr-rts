"""Fallback automatico para o modo so-placa quando o detector de objetos falha.

Regressao do modo de falha silencioso relatado em uso: com pesos YOLO presentes,
um video cujos veiculos o detector COCO nao reconhece (sintetico, animacao,
termico) produzia placas detectadas mas nenhuma trilha a que associa-las - e a
interface ficava vazia sem explicacao alguma.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.capture.source import VideoSource
from lprts.config import AppConfig
from lprts.core.pipeline import LPRPipeline
from lprts.detect.base import Detector
from lprts.detect.classic_plate import ClassicPlateDetector
from lprts.ocr.engines import build_ocr_engine


class DetectorCego(Detector):
    """Detector que nunca encontra nada - simula COCO sobre video estilizado."""

    name = "cego"

    def __init__(self) -> None:
        self.chamadas = 0

    def detect(self, image):
        self.chamadas += 1
        return []


class DetectorQueVe(Detector):
    """Detector que sempre devolve um veiculo cobrindo o quadro."""

    name = "vidente"

    def detect(self, image):
        from lprts.core.types import Detection

        h, w = image.shape[:2]
        return [Detection((0.0, 0.0, float(w), float(h)), 0.9, "car", 2)]


@pytest.fixture
def video_com_placa(tmp_path, plate_image):
    caminho = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    assert w.isOpened()
    for i in range(140):
        cena = np.full((360, 640, 3), 95, dtype=np.uint8)
        largura = 120 + (i % 60) * 2
        altura = int(largura * 0.33)
        cena[150 : 150 + altura, 250 : 250 + largura] = cv2.resize(
            plate_image("ABC1D23"), (largura, altura)
        )
        w.write(cena)
    w.release()
    return caminho


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.ocr.min_plate_width = 55
    cfg.ocr.ocr_every_n = 1
    cfg.analytics.save_crops = False
    cfg.detector.fallback_after_frames = 20
    return cfg


def _rodar(cfg, caminho, detector_objetos, store=None):
    with VideoSource(caminho) as src:
        pipe = LPRPipeline(
            cfg, detector_objetos, ClassicPlateDetector(min_height=12),
            build_ocr_engine("template"), store,
        ).start()
        placas = set()
        for r in pipe.run(src):
            for t in r.tracks:
                if t.plate_stable and t.plate_text:
                    placas.add(t.plate_text)
        pipe.finalize()
        ativo = pipe.fallback_active
        metricas = pipe.metrics.snapshot()
        pipe.stop()
    return placas, ativo, metricas


class TestFallbackAutomatico:
    def test_detector_cego_aciona_o_fallback_e_le_a_placa(self, tmp_path, video_com_placa):
        """Sem o fallback, este cenario produzia zero leituras."""
        placas, ativo, m = _rodar(_config(tmp_path), video_com_placa, DetectorCego())
        assert ativo is True
        assert m.n_reads > 0
        assert "ABC1D23" in placas

    def test_detector_que_funciona_nao_aciona_o_fallback(self, tmp_path, video_com_placa):
        placas, ativo, m = _rodar(_config(tmp_path), video_com_placa, DetectorQueVe())
        assert ativo is False
        assert m.n_reads > 0          # o caminho normal continua funcionando

    def test_fallback_pode_ser_desligado(self, tmp_path, video_com_placa):
        """configs/avaliacao.yaml desliga para manter o experimento deterministico."""
        cfg = _config(tmp_path)
        cfg.detector.auto_plate_fallback = False
        _, ativo, _ = _rodar(cfg, video_com_placa, DetectorCego())
        assert ativo is False

    def test_sem_placas_no_quadro_nao_aciona(self, tmp_path):
        """Uma via vazia nao deve disparar o fallback so por nao ter veiculos."""
        caminho = tmp_path / "vazio.mp4"
        w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
        for _ in range(80):
            w.write(np.full((360, 640, 3), 95, dtype=np.uint8))
        w.release()
        _, ativo, _ = _rodar(_config(tmp_path), caminho, DetectorCego())
        assert ativo is False

    def test_modo_so_placa_explicito_nao_usa_o_fallback(self, tmp_path, video_com_placa):
        cfg = _config(tmp_path)
        cfg.detector.detect_vehicles = False
        _, ativo, m = _rodar(cfg, video_com_placa, None)
        assert ativo is False          # ja era so-placa desde o inicio
        assert m.n_reads > 0

    def test_fallback_e_registrado_no_banco(self, tmp_path, video_com_placa):
        """Sem o registro, a config gravada descreveria outro comportamento."""
        cfg = _config(tmp_path)
        cfg.analytics.db_path = str(tmp_path / "s.db")
        with ResultStore(cfg.analytics.db_path) as store:
            store.start_session(str(video_com_placa))
            _rodar(cfg, video_com_placa, DetectorCego(), store)
            notas = store.query("SELECT notes FROM sessions")[0]["notes"] or ""
        assert "FALLBACK AUTOMATICO" in notas

    def test_nao_procura_placa_sem_veiculo_no_modo_normal(self, tmp_path):
        """Trabalho descartavel: essas placas nunca teriam trilha a que associar."""
        cfg = _config(tmp_path)
        cfg.detector.auto_plate_fallback = False
        pipe = LPRPipeline(
            cfg, DetectorCego(), ClassicPlateDetector(), build_ocr_engine("template"), None
        ).start()
        imagem = np.full((360, 640, 3), 95, dtype=np.uint8)
        assert pipe._detect_plates(imagem, vehicles=[]) == []
        pipe.stop()


class TestPesosForaDaPastaDoProjeto:
    """Abrir o app de outra pasta nao pode trocar os modelos em silencio."""

    def test_caminho_relativo_cai_na_raiz_do_projeto(self, tmp_path, monkeypatch):
        from lprts import factory

        raiz = tmp_path / "projeto"
        (raiz / "data" / "models").mkdir(parents=True)
        (raiz / "data" / "models" / "placa.pt").write_bytes(b"pesos")
        monkeypatch.setattr(factory, "_RAIZ_DO_PROJETO", raiz)
        outra = tmp_path / "outra_pasta"
        outra.mkdir()
        monkeypatch.chdir(outra)
        assert factory.resolve_weights("data/models/placa.pt") == raiz / "data/models/placa.pt"

    def test_pasta_atual_e_caminho_absoluto_tem_precedencia(self, tmp_path, monkeypatch):
        from lprts import factory

        monkeypatch.setattr(factory, "_RAIZ_DO_PROJETO", tmp_path / "projeto")
        (tmp_path / "data" / "models").mkdir(parents=True)
        (tmp_path / "data" / "models" / "placa.pt").write_bytes(b"local")
        monkeypatch.chdir(tmp_path)
        # Existe a partir daqui: fica como esta (a sessao grava o mesmo caminho de antes).
        assert factory.resolve_weights("data/models/placa.pt") == Path("data/models/placa.pt")
        absoluto = tmp_path / "x.pt"
        assert factory.resolve_weights(str(absoluto)) == absoluto
        # Inexistente nos dois lugares: devolve o pedido, e o aviso de sempre sai.
        assert factory.resolve_weights("data/models/nada.pt") == Path("data/models/nada.pt")
