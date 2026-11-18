"""Adaptadores de OCR: formato de entrada, visibilidade de falhas e limpeza.

Regressao do defeito que produzia "nenhuma placa reconhecida" sem causa visivel:
o adaptador do fast-plate-ocr desempacotava o retorno como tupla
``(textos, probabilidades)`` quando a API devolve ``list[PlatePrediction]``, e
passava a imagem em escala de cinza a um modelo que exige 3 canais. Toda leitura
levantava excecao - engolida em nivel DEBUG.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from lprts.ocr.base import OCREngine, empty_read
from lprts.ocr.engines import _garantir_3_canais, build_ocr_engine

TEM_FASTPLATE = importlib.util.find_spec("fast_plate_ocr") is not None


def placa(texto: str = "ABC1D23", w: int = 300, h: int = 100) -> np.ndarray:
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), 3)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(texto, fonte, 1.0, 2)
    escala = max(0.4, w * 0.84 / max(tw, 1))
    esp = max(2, int(round(escala * 2.6)))
    (tw, th), _ = cv2.getTextSize(texto, fonte, escala, esp)
    cv2.putText(img, texto, ((w - tw) // 2, (h + th) // 2), fonte, escala,
                (20, 20, 20), esp, cv2.LINE_AA)
    return img


class TestGarantir3Canais:
    def test_converte_escala_de_cinza(self):
        cinza = np.zeros((64, 128), dtype=np.uint8)
        assert _garantir_3_canais(cinza).shape == (64, 128, 3)

    def test_preserva_bgr(self):
        bgr = np.zeros((64, 128, 3), dtype=np.uint8)
        assert _garantir_3_canais(bgr).shape == (64, 128, 3)

    def test_converte_bgra(self):
        bgra = np.zeros((64, 128, 4), dtype=np.uint8)
        assert _garantir_3_canais(bgra).shape == (64, 128, 3)


class MotorQueFalha(OCREngine):
    name = "quebrado"

    def read(self, plate_crop):
        try:
            raise RuntimeError("modelo incompativel")
        except RuntimeError as exc:
            self._registrar_falha(exc)
            return empty_read(self.name)


class TestVisibilidadeDeFalhas:
    def test_conta_falhas_consecutivas(self):
        m = MotorQueFalha()
        for _ in range(3):
            m.read(placa())
        assert m.falhas_consecutivas == 3
        assert "modelo incompativel" in m.ultimo_erro

    def test_avisa_quando_a_falha_deixa_de_ser_esporadica(self, caplog):
        """Falhar sempre e defeito; falhar as vezes e a vida."""
        import logging

        m = MotorQueFalha()
        with caplog.at_level(logging.WARNING):
            for _ in range(OCREngine.FALHAS_ATE_AVISAR):
                m.read(placa())
        # getMessage() aplica os argumentos; .message so existe apos a formatacao.
        assert any("falhou em" in r.getMessage() for r in caplog.records)

    def test_sucesso_zera_o_contador(self):
        m = MotorQueFalha()
        m.read(placa())
        m._sucesso()
        assert m.falhas_consecutivas == 0

    def test_falha_nunca_propaga(self):
        """Uma leitura ruim nao pode derrubar o pipeline."""
        assert MotorQueFalha().read(placa()).text == ""


class TestLimpeza:
    def test_close_libera_o_modelo(self):
        motor = build_ocr_engine("template")
        motor._model = object()
        motor.close()
        assert motor._model is None

    def test_leitura_apos_close_nao_derruba(self):
        motor = build_ocr_engine("template")
        motor.close()
        motor.read(placa())      # nao deve levantar


@pytest.fixture(scope="module")
def motor():
    """Carregado uma vez por modulo: instanciar o modelo a cada teste e lento."""
    return build_ocr_engine("fastplate")


@pytest.mark.skipif(not TEM_FASTPLATE, reason="fast-plate-ocr nao instalado")
class TestFastPlateOCR:

    def test_le_placa_nitida(self, motor):
        r = motor.read(placa("ABC1D23"))
        assert r.text == "ABC1D23"
        assert r.valid_syntax
        assert r.confidence > 0.5

    def test_aceita_entrada_em_escala_de_cinza(self, motor):
        """Regressao: entrada 2D fazia o ONNX Runtime rejeitar as dimensoes."""
        cinza = cv2.cvtColor(placa("XYZ9K88"), cv2.COLOR_BGR2GRAY)
        assert motor.read(cinza).text == "XYZ9K88"

    def test_nao_acumula_falhas_em_uso_normal(self, motor):
        for texto in ("ABC1D23", "QTA8G86", "MCR4U95"):
            motor.read(placa(texto))
        assert motor.falhas_consecutivas == 0

    def test_devolve_confianca_por_caractere(self, motor):
        r = motor.read(placa("ABC1D23"))
        assert len(r.char_confidences) >= 7

    def test_recorte_vazio(self, motor):
        assert motor.read(np.zeros((0, 0, 3), dtype=np.uint8)).text == ""
