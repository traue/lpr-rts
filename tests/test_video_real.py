"""Regressoes encontradas ao rodar o sistema em video real (iPhone, 4K, 2026-10).

Cada teste reproduz um defeito visto nos videos ``IMG_0833``/``IMG_0834``; o
docstring diz o que acontecia.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from lprts.analytics.store import ResultStore
from lprts.config import AppConfig
from lprts.core.pipeline import LPRPipeline
from lprts.core.types import Detection, Frame, PlateRead, TrackState
from lprts.detect.base import Detector, merge_vehicle_duplicates
from lprts.ocr import plates_br
from lprts.ocr.base import OCREngine

IMG = np.zeros((720, 1280, 3), dtype=np.uint8)


class CarroDuplicado(Detector):
    """O mesmo veiculo como "car" e "truck" (IoU ~0,98), como o YOLO faz no video real."""

    name = "carro_duplicado"

    def __init__(self) -> None:
        self.frame = 0

    def detect(self, image):
        x = 100 + 8 * self.frame
        return [Detection((x, 300.0, x + 300.0, 500.0), 0.9, "car", 2),
                Detection((x + 2, 302.0, x + 300.0, 500.0), 0.6, "truck", 7)]


class PlacaNoCarro(Detector):
    name = "placa_no_carro"

    def detect(self, image):
        h, w = image.shape[:2]
        return [Detection((w * 0.4, h * 0.7, w * 0.6, h * 0.8), 0.9, "plate", 0)]


class OCRFixo(OCREngine):
    name = "fixo"

    def read(self, plate_crop):
        c = plates_br.correct("ABC1D23")
        return PlateRead(text=c.text, confidence=0.95, layout=c.layout, valid_syntax=c.valid,
                         engine=self.name, raw_text="ABC1D23")


def _cfg(**kw) -> AppConfig:
    cfg = AppConfig()
    cfg.analytics.save_crops = False
    cfg.ocr.min_plate_width = 10
    cfg.ocr.ocr_every_n = 1
    cfg.ocr.max_reads_published = 0
    for k, v in kw.items():
        cfg.override(**{k: v})
    return cfg


class TestVeiculoDuplicado:
    def test_classes_diferentes_sobre_o_mesmo_veiculo_viram_uma_caixa(self):
        carro = Detection((0, 0, 100, 50), 0.9, "car", 2)
        caminhao = Detection((1, 1, 100, 50), 0.6, "truck", 7)
        assert merge_vehicle_duplicates([carro, caminhao]) == [carro]

    def test_piloto_sobre_a_moto_nao_e_duplicata(self):
        """Pessoa e moto se sobrepoem (IoU ~0,5) e sao objetos distintos."""
        moto = Detection((0, 0, 100, 100), 0.7, "motorcycle", 3)
        piloto = Detection((0, 0, 100, 100), 0.9, "person", 0)
        assert len(merge_vehicle_duplicates([moto, piloto])) == 2

    def test_veiculos_vizinhos_nao_sao_fundidos(self):
        a = Detection((0, 0, 100, 50), 0.9, "car", 2)
        b = Detection((60, 0, 160, 50), 0.8, "car", 2)
        assert len(merge_vehicle_duplicates([a, b])) == 2

    def test_zero_desliga(self):
        carro = Detection((0, 0, 100, 50), 0.9, "car", 2)
        caminhao = Detection((1, 1, 100, 50), 0.6, "truck", 7)
        assert len(merge_vehicle_duplicates([carro, caminhao], 0.0)) == 2

    @pytest.mark.parametrize("iou,eventos", [(0.7, 1), (0.0, 2)])
    def test_placa_publicada_uma_vez(self, tmp_path, iou, eventos):
        """Antes: cada caixa virava uma trilha e a mesma placa era publicada duas vezes."""
        db = tmp_path / f"d{iou}.db"
        det = CarroDuplicado()
        cfg = _cfg(**{"detector.vehicle_merge_iou": iou, "detector.classes": "veiculos"})
        with ResultStore(db) as store:
            store.start_session("s")
            pipe = LPRPipeline(cfg, det, PlacaNoCarro(), OCRFixo(), store).start()
            for i in range(20):
                det.frame = i
                pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=i / 30))
            pipe.finalize()
            pipe.stop()
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == eventos


class TestAtribuicaoDePlaca:
    def test_placa_vai_para_o_veiculo_centrado_nela(self):
        """Antes: o empate seguia a ordem da lista e mudava de quadro a quadro.

        Caixas de carros vizinhos se sobrepoem em perspectiva e as duas contem a
        placa por inteiro. A placa traseira e centrada no proprio veiculo; a
        caixa menor (A) nao e criterio - aqui ela pertence a B.
        """
        pipe = LPRPipeline(_cfg(), CarroDuplicado(), None, None, None)
        from lprts.core.taxonomy import ObjectClass

        a = TrackState(track_id=1, bbox=(0.0, 0.0, 800.0, 500.0), score=0.9,
                       object_class=ObjectClass.CARRO)
        b = TrackState(track_id=2, bbox=(500.0, 0.0, 1500.0, 500.0), score=0.9,
                       object_class=ObjectClass.CARRO)
        placa = Detection((900.0, 400.0, 1000.0, 430.0), 0.9, "plate", 0)
        for ordem in ([a, b], [b, a]):
            atribuicao = pipe._assign_plates([placa], ordem)
            assert list(atribuicao) == [2]

    def test_pessoa_e_bicicleta_nao_recebem_placa(self):
        """Antes: a caixa do piloto contem a placa da moto e podia ficar com ela -
        a trilha da pessoa nunca passa pelo OCR, e a moto ficava sem leitura."""
        from lprts.core.taxonomy import ObjectClass

        pipe = LPRPipeline(_cfg(), CarroDuplicado(), None, None, None)
        moto = TrackState(track_id=1, bbox=(400.0, 300.0, 600.0, 700.0), score=0.9,
                          object_class=ObjectClass.MOTOCICLETA)
        piloto = TrackState(track_id=2, bbox=(430.0, 100.0, 570.0, 650.0), score=0.9,
                            object_class=ObjectClass.PESSOA)
        placa = Detection((470.0, 560.0, 530.0, 600.0), 0.9, "plate", 0)
        for ordem in ([moto, piloto], [piloto, moto]):
            assert list(pipe._assign_plates([placa], ordem)) == [1]


class TestFonteSintetica:
    def test_reconhece_o_video_do_gerador(self, tmp_path):
        from lprts.capture.devices import is_synthetic_video

        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        assert not is_synthetic_video(str(video))
        (tmp_path / "v_meta.json").write_text(
            json.dumps({"generator": "scripts/make_synthetic_dataset.py"}), encoding="utf-8")
        assert is_synthetic_video(str(video))
        assert not is_synthetic_video("0") and not is_synthetic_video("rtsp://cam/x")


def _duas_sessoes(db: Path) -> list[str]:
    """Dois 'videos' no mesmo banco, como acontece depois de algumas execucoes."""
    sessoes = []
    for nome in ("primeiro.MOV", "segundo.MOV"):
        det = CarroDuplicado()
        cfg = _cfg(**{"detector.classes": "veiculos"})
        cfg.source.uri = f"/videos/{nome}"
        with ResultStore(db) as store:
            store.start_session(cfg.source.uri)
            pipe = LPRPipeline(cfg, det, PlacaNoCarro(), OCRFixo(), store).start()
            for i in range(12):
                det.frame = i
                pipe.process_frame(Frame(i, IMG, timestamp=0.0, source_pts=i / 30))
            pipe.finalize()
            pipe.stop()
            sessoes.append(store.session_id)
    return sessoes


class TestExportacaoDoVideoAnalisado:
    def test_exporta_so_a_sessao(self, tmp_path):
        """Antes: o Exportar levava o banco inteiro - todos os videos ja processados."""
        from lprts.analytics.export import export_analyzed_video

        db = tmp_path / "r.db"
        primeiro, segundo = _duas_sessoes(db)
        sid, pasta, arquivos = export_analyzed_video(db, tmp_path / "exp")
        assert sid == segundo                                   # padrao: a mais recente
        assert pasta.name == f"segundo_{segundo}"
        assert (pasta / "sessao.json").exists()
        for arq in arquivos:
            if arq.suffix != ".csv":
                continue
            with arq.open(encoding="utf-8") as fh:
                sessoes = {linha["session_id"] for linha in csv.DictReader(fh)}
            assert sessoes <= {segundo}, arq.name
        assert export_analyzed_video(db, tmp_path / "exp", primeiro)[1].name == f"primeiro_{primeiro}"

    def test_cli_exporta_o_ultimo_video_e_all_exporta_tudo(self, tmp_path):
        from typer.testing import CliRunner

        from lprts.cli import app

        db = tmp_path / "r.db"
        _, segundo = _duas_sessoes(db)
        r = CliRunner().invoke(app, ["export", "--db", str(db), "--out", str(tmp_path / "e")])
        assert r.exit_code == 0, r.output
        assert [p.name for p in (tmp_path / "e").iterdir()] == [f"segundo_{segundo}"]
        r = CliRunner().invoke(app, ["export", "--db", str(db), "--out", str(tmp_path / "t"), "--all"])
        assert r.exit_code == 0, r.output
        with (tmp_path / "t" / "sessions.csv").open(encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 2


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6", reason="interface grafica nao instalada")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


class TestInterface:
    def test_padroes_por_fonte(self, qapp, tmp_path):
        from lprts.ui.app import padroes_para_fonte

        (tmp_path / "s.mp4").write_bytes(b"x")
        (tmp_path / "s_meta.json").write_text('{"generator": "make_synthetic_dataset"}')
        assert padroes_para_fonte(str(tmp_path / "s.mp4")) == (False, "classic")
        assert padroes_para_fonte(str(tmp_path / "IMG_0834.MOV")) == (True, "auto")
        assert padroes_para_fonte("0") == (True, "auto")

    def test_video_real_aberto_depois_do_sintetico_liga_o_yolo(self, qapp, tmp_path):
        """Antes: o ./start.sh abria a demo com --no-vehicle-detector e um video real
        aberto depois na interface rodava sem o YOLO."""
        import importlib.util

        from lprts.ui.app import MainWindow

        if importlib.util.find_spec("ultralytics") is None:
            pytest.skip("ultralytics nao instalado: a caixa fica desabilitada")
        (tmp_path / "s.mp4").write_bytes(b"x")
        (tmp_path / "s_meta.json").write_text('{"generator": "make_synthetic_dataset"}')
        pesos = tmp_path / "yolo.pt"
        pesos.write_bytes(b"x")
        cfg = AppConfig()
        cfg.source.uri = str(tmp_path / "s.mp4")
        cfg.detector.vehicle_weights = str(pesos)
        cfg.detector.detect_vehicles = False          # o que o ./start.sh passa na demo
        cfg.detector.plate_detector = "classic"
        janela = MainWindow(cfg)
        try:
            assert not janela.vehicle_check.isChecked()
            janela.source_selector.set_uri(str(tmp_path / "IMG_0834.MOV"))
            assert janela.vehicle_check.isChecked()
            janela._apply_controls()
            assert cfg.detector.detect_vehicles and cfg.detector.plate_detector == "auto"
            janela.source_selector.set_uri(str(tmp_path / "s.mp4"))
            assert not janela.vehicle_check.isChecked()
        finally:
            janela.close()


def test_importar_ultralytics_nao_deixa_o_opencv_numa_thread(tmp_path, monkeypatch):
    """Antes: importar o ultralytics fazia cv2.setNumThreads(0) no processo inteiro.

    Num video 4K vertical, o redimensionamento passou de ~7 ms para 70 ms por
    quadro e o processamento caiu para ~10 quadros/s so por ligar o YOLO.
    """
    import sys

    import cv2

    from lprts.runtime import import_ultralytics

    falso = tmp_path / "ultralytics"
    falso.mkdir()
    (falso / "__init__.py").write_text(
        "import cv2\ncv2.setNumThreads(0)\n__version__ = 'falso'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    antes = cv2.getNumThreads()
    if antes <= 1:
        pytest.skip("OpenCV ja esta sequencial nesta maquina")
    try:
        assert import_ultralytics().__version__ == "falso"
        # No macOS (backend GCD) o getter so distingue paralelo de sequencial;
        # a propriedade que importa vale em qualquer backend: o estado volta.
        assert cv2.getNumThreads() == antes
    finally:
        cv2.setNumThreads(antes)
