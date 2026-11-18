"""Gramatica de placas brasileiras: normalizacao, validacao e correcao."""

from __future__ import annotations

import pytest

from lprts.core.types import PlateLayout
from lprts.ocr import plates_br


class TestNormalizacao:
    @pytest.mark.parametrize(
        "entrada,esperado",
        [
            ("abc1d23", "ABC1D23"),
            ("ABC-1234", "ABC1234"),
            ("ABC 1D23", "ABC1D23"),
            ("BRASIL ABC1D23", "ABC1D23"),
            ("MERCOSULABC1D23", "ABC1D23"),
            ("", ""),
        ],
    )
    def test_normaliza(self, entrada, esperado):
        assert plates_br.normalize(entrada) == esperado


class TestLayout:
    def test_mercosul(self):
        assert plates_br.detect_layout("ABC1D23") is PlateLayout.MERCOSUL

    def test_antigo(self):
        assert plates_br.detect_layout("ABC1234") is PlateLayout.LEGACY

    @pytest.mark.parametrize("texto", ["ABCD123", "12345678", "ABC12", "", "AB1C234"])
    def test_invalidos(self, texto):
        assert plates_br.detect_layout(texto) is PlateLayout.UNKNOWN
        assert not plates_br.is_valid(texto)


class TestCorrecao:
    @pytest.mark.parametrize(
        "entrada,esperado,correcoes",
        [
            ("ABC1D23", "ABC1D23", 0),   # ja valida: nao mexe
            ("ABC1234", "ABC1234", 0),
            ("4BC1D23", "ABC1D23", 1),   # digito onde deveria haver letra
            ("ABCID23", "ABC1D23", 1),   # letra onde deveria haver digito
            ("ABCI234", "ABC1234", 1),
            ("A8C1D23", "ABC1D23", 1),
            ("ABC1DZ3", "ABC1D23", 1),
        ],
    )
    def test_corrige_pela_mascara(self, entrada, esperado, correcoes):
        resultado = plates_br.correct(entrada)
        assert resultado.text == esperado
        assert resultado.valid
        assert resultado.corrections == correcoes

    def test_nao_inventa_placa(self):
        """Correcoes demais indicam OCR ruim, nao placa a recuperar."""
        assert not plates_br.correct("1231231", max_corrections=1).valid

    def test_comprimento_errado_e_invalido(self):
        for texto in ("ABC12345", "ABC123", "", "A"):
            assert not plates_br.correct(texto).valid

    def test_letra_valida_na_posicao_nao_e_alterada(self):
        """O na posicao 5 do Mercosul e letra legitima - nao virar zero."""
        assert plates_br.correct("ABC1O23").text == "ABC1O23"


class TestSalvage:
    @pytest.mark.parametrize(
        "entrada,esperado",
        [
            ("BRASILABC1D23", "ABC1D23"),
            ("XXABC1D23YY", "ABC1D23"),
            ("ABC1D23", "ABC1D23"),
        ],
    )
    def test_extrai_janela_valida(self, entrada, esperado):
        assert plates_br.salvage(entrada) == esperado

    def test_texto_curto_passa_direto(self):
        assert plates_br.salvage("ABC") == "ABC"


class TestAlfabetoPosicional:
    def test_mercosul_posicoes(self):
        for pos in (0, 1, 2, 4):
            assert plates_br.position_alphabet(pos, PlateLayout.MERCOSUL).isalpha()
        for pos in (3, 5, 6):
            assert plates_br.position_alphabet(pos, PlateLayout.MERCOSUL).isdigit()

    def test_antigo_posicoes(self):
        for pos in (3, 4, 5, 6):
            assert plates_br.position_alphabet(pos, PlateLayout.LEGACY).isdigit()
