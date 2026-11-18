"""Fusao multi-frame: consolida varias leituras de uma mesma trilha numa placa.

Motivacao. Um unico quadro e uma amostra ruidosa: desfoque de movimento, um
reflexo ou um angulo ruim erram um caractere. Como o rastreador garante que
todas as leituras de um ``track_id`` vem do *mesmo veiculo*, podemos combina-las.

Duas votacoes complementares, ambas ponderadas:

* **por caractere** (padrao): vota-se a posicao 0, a 1, ... independentemente.
  E mais forte porque aproveita quadros parcialmente corretos - o quadro A pode
  acertar os 3 primeiros caracteres e o quadro B os 4 ultimos.
* **por string inteira**: conta a placa completa mais frequente. Mais
  conservador; util quando os textos tem comprimentos diferentes.

Peso de cada leitura = confianca do OCR x nitidez relativa x bonus de sintaxe
valida. A nitidez entra porque um quadro nitido erra menos, e o bonus de
sintaxe porque uma leitura que fecha a gramatica brasileira e evidencia forte.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.types import PlateLayout, PlateRead, TrackState
from ..ocr import plates_br


@dataclass(slots=True)
class FusedPlate:
    """Resultado consolidado de uma trilha."""

    text: str
    confidence: float
    layout: PlateLayout
    valid_syntax: bool
    n_reads: int = 0
    agreement: float = 0.0          # fracao de votos recebidos pelo vencedor
    per_char_confidence: list[float] = field(default_factory=list)
    stable: bool = False            # atende aos criterios minimos de publicacao

    @property
    def formatted(self) -> str:
        if self.layout is PlateLayout.LEGACY and len(self.text) == plates_br.PLATE_LENGTH:
            return f"{self.text[:3]}-{self.text[3:]}"
        return self.text


def read_weight(read: PlateRead, max_sharpness: float = 1.0, syntax_bonus: float = 1.35) -> float:
    """Peso do voto de uma leitura.

    A nitidez entra em raiz quadrada para nao dominar: e um indicio de qualidade,
    nao uma medida de correcao.
    """
    if not read.text:
        return 0.0
    weight = max(float(read.confidence), 1e-3)
    if max_sharpness > 0 and read.sharpness > 0:
        weight *= math.sqrt(min(1.0, read.sharpness / max_sharpness) + 1e-6) + 0.5
    if read.valid_syntax:
        weight *= syntax_bonus
    return weight


def vote_per_character(
    reads: list[PlateRead],
    expected_length: int = plates_br.PLATE_LENGTH,
    max_corrections: int = plates_br.DEFAULT_MAX_CORRECTIONS,
) -> FusedPlate:
    """Votacao ponderada, posicao a posicao, entre leituras do mesmo comprimento."""
    usable = [r for r in reads if r.text and len(r.text) == expected_length]
    if not usable:
        return vote_per_string(reads, max_corrections=max_corrections)

    max_sharp = max((r.sharpness for r in usable), default=0.0)
    weights = [read_weight(r, max_sharp) for r in usable]
    total_weight = sum(weights)
    if total_weight <= 0:
        return vote_per_string(reads, max_corrections=max_corrections)

    chars: list[str] = []
    char_conf: list[float] = []
    agreements: list[float] = []

    for pos in range(expected_length):
        tally: dict[str, float] = defaultdict(float)
        for read, weight in zip(usable, weights, strict=True):
            ch = read.text[pos]
            # Se o motor deu confianca por caractere, usa-a nesta posicao.
            if len(read.char_confidences) == expected_length:
                tally[ch] += weight * max(read.char_confidences[pos], 1e-3)
            else:
                tally[ch] += weight
        best_char = max(tally, key=lambda c: tally[c])
        pos_total = sum(tally.values())
        chars.append(best_char)
        share = tally[best_char] / pos_total if pos_total > 0 else 0.0
        char_conf.append(share)
        agreements.append(share)

    voted = "".join(chars)
    candidate = plates_br.correct(voted, max_corrections=max_corrections)
    confidence = (sum(char_conf) / len(char_conf)) * (
        sum(r.confidence * w for r, w in zip(usable, weights, strict=True)) / total_weight
    )
    return FusedPlate(
        text=candidate.text or voted,
        confidence=float(confidence),
        layout=candidate.layout,
        valid_syntax=candidate.valid,
        n_reads=len(usable),
        agreement=float(sum(agreements) / len(agreements)),
        per_char_confidence=char_conf,
    )


def vote_per_string(
    reads: list[PlateRead], max_corrections: int = plates_br.DEFAULT_MAX_CORRECTIONS
) -> FusedPlate:
    """Votacao ponderada pela string completa (fallback e modo conservador)."""
    usable = [r for r in reads if r.text]
    if not usable:
        return FusedPlate("", 0.0, PlateLayout.UNKNOWN, False, 0, 0.0)

    max_sharp = max((r.sharpness for r in usable), default=0.0)
    tally: dict[str, float] = defaultdict(float)
    conf_sum: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for read in usable:
        weight = read_weight(read, max_sharp)
        tally[read.text] += weight
        conf_sum[read.text] += read.confidence
        counts[read.text] += 1

    total = sum(tally.values())
    best = max(tally, key=lambda t: tally[t])
    candidate = plates_br.correct(best, max_corrections=max_corrections)
    return FusedPlate(
        text=candidate.text or best,
        confidence=float(conf_sum[best] / max(counts[best], 1)),
        layout=candidate.layout,
        valid_syntax=candidate.valid,
        n_reads=len(usable),
        agreement=float(tally[best] / total) if total > 0 else 0.0,
    )


class PlateVoter:
    """Aplica a fusao a uma trilha e decide quando o resultado pode ser publicado.

    "Publicar" cedo demais gera placas erradas no banco; tarde demais perde o
    veiculo. Os tres criterios (numero minimo de leituras, concordancia minima e
    sintaxe valida) sao configuraveis para que o compromisso possa ser estudado
    empiricamente - e um dos parametros interessantes de varrer numa tese.
    """

    def __init__(
        self,
        min_reads: int = 3,
        min_agreement: float = 0.6,
        min_confidence: float = 0.45,
        require_valid_syntax: bool = True,
        mode: str = "char",
        max_reads_per_track: int = 60,
        max_corrections: int = plates_br.DEFAULT_MAX_CORRECTIONS,
    ) -> None:
        self.min_reads = min_reads
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.require_valid_syntax = require_valid_syntax
        self.mode = mode
        # 0 = sem limite. Avaliacao deve usar 0: o corte por confianca muda o
        # resultado da votacao e impede que a refusao offline reproduza o online.
        self.max_reads_per_track = max_reads_per_track
        self.max_corrections = max_corrections

    def add_read(self, track: TrackState, read: PlateRead) -> None:
        """Registra uma leitura na trilha, limitando o historico se configurado."""
        if not read.text:
            return
        track.reads.append(read)
        track.n_reads_total += 1
        if self.max_reads_per_track and len(track.reads) > self.max_reads_per_track:
            # Mantem as leituras de maior confianca: sao as que mais informam.
            track.reads.sort(key=lambda r: r.confidence, reverse=True)
            del track.reads[self.max_reads_per_track :]

    def vote(self, reads: list[PlateRead]) -> FusedPlate:
        """Aplica a votacao configurada e o criterio de estabilidade, sem tocar em trilha."""
        fused = (
            vote_per_character(reads, max_corrections=self.max_corrections)
            if self.mode == "char"
            else vote_per_string(reads, max_corrections=self.max_corrections)
        )
        fused.stable = self.is_stable(fused)
        return fused

    def fuse(self, track: TrackState) -> FusedPlate:
        fused = self.vote(track.reads)
        # Antes da publicacao, o texto provisorio acompanha a melhor estimativa
        # corrente. (Uma versao anterior o congelava no primeiro fragmento lido,
        # e era esse fragmento que ia para o banco no fim do video.) Depois da
        # publicacao, so uma nova fusao estavel o substitui: leituras ruins de um
        # veiculo que se afasta nao rebaixam uma placa ja confirmada.
        if fused.stable or not track.plate_stable:
            track.plate_text = fused.text or None
            track.plate_confidence = fused.confidence
            track.plate_layout = fused.layout
        track.plate_stable = track.plate_stable or fused.stable
        return fused

    def is_stable(self, fused: FusedPlate) -> bool:
        if fused.n_reads < self.min_reads:
            return False
        if fused.agreement < self.min_agreement:
            return False
        if fused.confidence < self.min_confidence:
            return False
        if self.require_valid_syntax and not fused.valid_syntax:
            return False
        return True
