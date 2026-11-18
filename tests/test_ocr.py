"""OCR classico: segmentacao, casamento de gabaritos e decodificacao restrita."""

from __future__ import annotations

import numpy as np
import pytest

from lprts.ocr import engines
from lprts.ocr.engines import EngineUnavailableError, available_engines, build_ocr_engine
from lprts.ocr.template_ocr import TemplateOCR, segment_characters
from lprts.utils.imgproc import enhance_plate, sharpness


class TestSegmentacao:
    def test_encontra_sete_caracteres(self, plate_image):
        chars = segment_characters(enhance_plate(plate_image("ABC1D23"), 64))
        assert len(chars) == 7

    def test_robusto_a_margem_do_recorte(self, plate_image):
        """O agrupamento por altura mediana nao pode depender da margem."""
        apertado = segment_characters(enhance_plate(plate_image("ABC1D23", 200, 60), 64))
        folgado = segment_characters(enhance_plate(plate_image("ABC1D23", 420, 160), 64))
        assert len(apertado) == len(folgado) == 7

    def test_imagem_vazia(self):
        assert segment_characters(np.zeros((0, 0), dtype=np.uint8)) == []

    def test_imagem_uniforme_nao_gera_caracteres(self):
        assert segment_characters(np.full((64, 200), 200, dtype=np.uint8)) == []

    def test_devolve_mascara_binaria(self, plate_image):
        """Polaridade igual a dos gabaritos: caractere branco sobre preto."""
        chars = segment_characters(enhance_plate(plate_image("ABC1D23"), 64))
        assert all(set(np.unique(c)).issubset({0, 255}) for c in chars)


class TestTemplateOCR:
    @pytest.mark.parametrize(
        "texto", ["ABC1D23", "XYZ9K88", "ABC1234", "QWE5R66", "MNO7P41"]
    )
    def test_le_placas_nitidas(self, plate_image, texto):
        r = TemplateOCR().read(plate_image(texto))
        assert r.text == texto
        assert r.valid_syntax
        assert r.confidence > 0.6

    @pytest.mark.parametrize("blur", [3, 5])
    def test_tolera_desfoque_moderado(self, plate_image, blur):
        assert TemplateOCR().read(plate_image("ABC1D23", blur=blur)).text == "ABC1D23"

    def test_recorte_vazio(self):
        r = TemplateOCR().read(np.zeros((0, 0, 3), dtype=np.uint8))
        assert r.text == "" and r.confidence == 0.0

    def test_ruido_nao_produz_placa_valida(self):
        rng = np.random.default_rng(0)
        ruido = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
        assert not TemplateOCR().read(ruido).valid_syntax

    def test_decodificacao_respeita_a_mascara(self, plate_image):
        """As posicoes de letra nunca podem receber digito, e vice-versa."""
        r = TemplateOCR().read(plate_image("ABC1D23"))
        assert r.text[:3].isalpha() and r.text[3].isdigit() and r.text[5:].isdigit()

    def test_reporta_tempo(self, plate_image):
        assert TemplateOCR().read(plate_image()).ocr_ms > 0


class TestFabrica:
    def test_template_sempre_disponivel(self):
        assert "template" in available_engines()

    def test_auto_sempre_devolve_um_motor(self):
        assert build_ocr_engine("auto") is not None

    def test_motor_explicito_ausente_informa_o_erro(self, monkeypatch):
        def indisponivel(**kwargs):
            raise ImportError("dependencia ausente")

        monkeypatch.setitem(engines._ENGINES, "fastplate", indisponivel)
        with pytest.raises(EngineUnavailableError, match="explicitamente"):
            build_ocr_engine("fastplate")

    def test_auto_usa_baseline_quando_motores_estao_ausentes(self, monkeypatch):
        def indisponivel(**kwargs):
            raise ImportError("dependencia ausente")

        for name in ("fastplate", "paddle", "easyocr"):
            monkeypatch.setitem(engines._ENGINES, name, indisponivel)
        assert build_ocr_engine("auto").name == "template"

    def test_nome_invalido_levanta(self):
        with pytest.raises(ValueError):
            build_ocr_engine("motor_inexistente")


class TestQualidadeDeImagem:
    def test_nitidez_discrimina_desfoque(self, plate_image):
        assert sharpness(plate_image("ABC1D23")) > sharpness(plate_image("ABC1D23", blur=9))

    def test_nitidez_de_imagem_vazia(self):
        assert sharpness(np.zeros((0, 0), dtype=np.uint8)) == 0.0
