"""Fusao multi-frame: votacao por caractere e criterio de estabilidade."""

from __future__ import annotations

from lprts.core.types import PlateRead, TrackState
from lprts.fusion.voting import PlateVoter, vote_per_character, vote_per_string
from lprts.ocr.plates_br import correct


def leitura(texto: str, conf: float = 0.8, nitidez: float = 100.0) -> PlateRead:
    c = correct(texto)
    return PlateRead(
        text=texto, confidence=conf, layout=c.layout, valid_syntax=c.valid, sharpness=nitidez
    )


class TestVotacaoPorCaractere:
    def test_maioria_vence(self):
        leituras = [leitura("ABC1D23"), leitura("ABC1D23"), leitura("ABC1D28")]
        assert vote_per_character(leituras).text == "ABC1D23"

    def test_combina_quadros_parcialmente_corretos(self):
        """Nenhuma leitura esta inteira certa, mas a votacao posicional recupera."""
        leituras = [
            leitura("ABC1D99", 0.7),   # acerta os 5 primeiros
            leitura("XYZ1D23", 0.7),   # acerta os 4 ultimos
            leitura("ABC1D23", 0.9),
        ]
        assert vote_per_character(leituras).text == "ABC1D23"

    def test_confianca_pondera_o_voto(self):
        leituras = [leitura("ABC1D23", 0.95, 300.0), leitura("XYZ9K88", 0.20, 10.0)]
        assert vote_per_character(leituras).text == "ABC1D23"

    def test_concordancia_total(self):
        r = vote_per_character([leitura("ABC1D23")] * 4)
        assert r.agreement == 1.0 and r.n_reads == 4

    def test_sem_leituras(self):
        r = vote_per_character([])
        assert r.text == "" and r.n_reads == 0


class TestVotacaoPorString:
    def test_placa_mais_frequente(self):
        leituras = [leitura("ABC1D23"), leitura("ABC1D23"), leitura("XYZ9K88", 0.3)]
        assert vote_per_string(leituras).text == "ABC1D23"

    def test_comprimentos_diferentes(self):
        r = vote_per_string([leitura("ABC1D23"), leitura("ABC1D2"), leitura("ABC1D23")])
        assert r.text == "ABC1D23"


class TestPlateVoter:
    def _trilha(self) -> TrackState:
        return TrackState(track_id=1, bbox=(0, 0, 100, 40), score=0.9)

    def test_publica_quando_estavel(self):
        voter = PlateVoter(min_reads=3, min_agreement=0.6)
        t = self._trilha()
        for _ in range(3):
            voter.add_read(t, leitura("ABC1D23", 0.9))
        r = voter.fuse(t)
        assert r.stable and t.plate_text == "ABC1D23" and t.plate_stable

    def test_nao_publica_com_poucas_leituras(self):
        voter = PlateVoter(min_reads=5)
        t = self._trilha()
        for _ in range(2):
            voter.add_read(t, leitura("ABC1D23", 0.9))
        assert not voter.fuse(t).stable

    def test_nao_publica_sintaxe_invalida(self):
        voter = PlateVoter(min_reads=1, require_valid_syntax=True)
        t = self._trilha()
        for _ in range(3):
            voter.add_read(t, PlateRead(text="ZIX", confidence=0.9))
        r = voter.fuse(t)
        assert not r.stable and not t.plate_stable

    def test_estabilidade_nao_regride(self):
        """Uma placa ja publicada nao pode ser rebaixada por leituras ruins."""
        voter = PlateVoter(min_reads=3, min_agreement=0.6)
        t = self._trilha()
        for _ in range(4):
            voter.add_read(t, leitura("ABC1D23", 0.95))
        assert voter.fuse(t).stable
        for _ in range(8):
            voter.add_read(t, leitura("XYZ9K88", 0.2, 5.0))
        voter.fuse(t)
        assert t.plate_stable

    def test_historico_limitado(self):
        voter = PlateVoter(max_reads_per_track=10)
        t = self._trilha()
        for i in range(50):
            voter.add_read(t, leitura("ABC1D23", 0.5 + i * 0.01))
        assert len(t.reads) <= 10

    def test_leitura_vazia_ignorada(self):
        voter = PlateVoter()
        t = self._trilha()
        voter.add_read(t, PlateRead(text="", confidence=0.0))
        assert len(t.reads) == 0
