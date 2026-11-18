"""Espaco de busca dos hiperparametros do pipeline.

Os parametros aqui nao sao pesos de rede - sao as decisoes de engenharia que
governam o compromisso entre exatidao e latencia. Sao poucos, interpretaveis e
cada um tem efeito conhecido, o que torna a fronteira de Pareto resultante
legivel: da para explicar *por que* cada ponto esta onde esta.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

ParamKind = Literal["float", "int", "categorical"]


@dataclass(slots=True)
class Param:
    """Um hiperparametro e seu dominio."""

    name: str                       # caminho pontilhado, ex.: "ocr.min_plate_width"
    kind: ParamKind
    low: float = 0.0
    high: float = 1.0
    step: float = 0.0
    choices: tuple[Any, ...] = ()
    log: bool = False
    description: str = ""

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "categorical":
            return rng.choice(self.choices)
        if self.kind == "int":
            passo = int(self.step) or 1
            n = int((self.high - self.low) // passo)
            return int(self.low) + passo * rng.randint(0, max(n, 0))
        valor = rng.uniform(self.low, self.high)
        return self._quantize(valor)

    def mutate(self, valor: Any, rng: random.Random, forca: float = 0.2) -> Any:
        """Perturbacao local, usada na busca evolutiva."""
        if self.kind == "categorical":
            return rng.choice(self.choices) if rng.random() < forca else valor
        faixa = self.high - self.low
        novo = float(valor) + rng.gauss(0.0, forca * faixa)
        novo = min(max(novo, self.low), self.high)
        if self.kind == "int":
            passo = int(self.step) or 1
            return int(round(novo / passo) * passo)
        return self._quantize(novo)

    def _quantize(self, valor: float) -> float:
        """Ajusta ao passo e arredonda.

        O arredondamento final nao e cosmetico: sem ele, ``0.9 + 0.05`` vira
        0.9500000000000001 e a mesma configuracao aparece como duas distintas na
        fronteira, poluindo o relatorio e inflando a contagem de trials.
        """
        if self.step > 0:
            valor = round(valor / self.step) * self.step
        valor = min(max(valor, self.low), self.high)
        casas = max(0, -int(round(__import__("math").log10(self.step)))) + 2 if self.step > 0 else 6
        return round(float(valor), casas)


@dataclass(slots=True)
class SearchSpace:
    """Conjunto de parametros a otimizar."""

    params: list[Param] = field(default_factory=list)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}

    def mutate(self, base: dict[str, Any], rng: random.Random, forca: float = 0.2) -> dict[str, Any]:
        return {p.name: p.mutate(base.get(p.name, p.sample(rng)), rng, forca) for p in self.params}

    def crossover(
        self, a: dict[str, Any], b: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Recombinacao uniforme entre duas configuracoes."""
        return {p.name: (a if rng.random() < 0.5 else b).get(p.name) for p in self.params}

    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def get(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def __len__(self) -> int:
        return len(self.params)


# ---------------------------------------------------------------------------
# Espacos prontos
# ---------------------------------------------------------------------------

def fusion_space() -> SearchSpace:
    """Parametros da fusao multi-frame.

    E o espaco mais valioso para varrer, porque a avaliacao e **barata**: as
    leituras brutas ja estao no banco (tabela ``reads``), entao cada tentativa
    custa milissegundos em vez de reprocessar o video. Milhares de tentativas
    ficam viaveis.
    """
    return SearchSpace([
        Param("fusion.mode", "categorical", choices=("char", "string"),
              description="votacao por caractere ou por placa inteira"),
        Param("fusion.min_reads", "int", 1, 12, 1,
              description="leituras minimas antes de publicar"),
        Param("fusion.min_agreement", "float", 0.30, 0.95, 0.05,
              description="concordancia minima entre as leituras"),
        Param("fusion.min_confidence", "float", 0.20, 0.90, 0.05,
              description="confianca minima da fusao"),
        Param("fusion.require_valid_syntax", "categorical", choices=(True, False),
              description="exigir que a placa feche a gramatica brasileira"),
        Param("fusion.publish_policy", "categorical",
              choices=("first_stable", "revise", "on_exit"),
              description="quando publicar: 1a fusao estavel, com revisao, ou na saida"),
        Param("fusion.exit_min_reads", "int", 0, 3, 1,
              description="leituras minimas na saida para trilha sem decisao (0 = desligado)"),
    ])


def realtime_space() -> SearchSpace:
    """Parametros que governam o compromisso exatidao x latencia.

    Cada avaliacao exige uma passada completa pelo video, entao o orcamento de
    tentativas e pequeno - dezenas, nao milhares.
    """
    return SearchSpace([
        Param("detector.imgsz", "categorical", choices=(320, 416, 512, 640),
              description="resolucao de entrada do detector"),
        Param("detector.conf", "float", 0.15, 0.60, 0.05,
              description="limiar de confianca da deteccao"),
        Param("detector.detect_every_n", "int", 1, 4, 1,
              description="detectar a cada N quadros (Kalman preenche o resto)"),
        Param("ocr.ocr_every_n", "int", 1, 6, 1,
              description="rodar OCR a cada N quadros por trilha"),
        Param("ocr.max_ocr_per_frame", "int", 1, 8, 1,
              description="teto de leituras por quadro"),
        Param("ocr.min_plate_width", "int", 40, 120, 10,
              description="largura minima da placa para tentar OCR"),
        Param("source.resize_width", "categorical", choices=(640, 960, 1280, 1920),
              description="largura de reamostragem da entrada"),
    ])


def tracker_space() -> SearchSpace:
    """Parametros de associacao do rastreador (limiares em IoU minimo)."""
    return SearchSpace([
        Param("tracker.match_iou", "float", 0.10, 0.80, 0.05,
              description="IoU minimo na 1a passada (ByteTrack de referencia: 0,2)"),
        Param("tracker.second_match_iou", "float", 0.20, 0.80, 0.05,
              description="IoU minimo na 2a passada (deteccoes fracas)"),
        Param("tracker.max_age", "int", 5, 60, 5, description="em quadros da fonte"),
        Param("tracker.min_hits", "int", 1, 6, 1),
        Param("tracker.plate_match_iou", "float", 0.15, 0.70, 0.05),
    ])


SPACES = {
    "fusion": fusion_space,
    "realtime": realtime_space,
    "tracker": tracker_space,
}


def build_space(name: str) -> SearchSpace:
    key = (name or "fusion").lower().strip()
    if key not in SPACES:
        raise ValueError(f"Espaco de busca desconhecido: {name!r}. Opcoes: {sorted(SPACES)}")
    return SPACES[key]()


def combined_space(*names: str) -> SearchSpace:
    """Une varios espacos (ex.: ``combined_space("fusion", "tracker")``)."""
    params: list[Param] = []
    vistos: set[str] = set()
    for nome in names:
        for p in build_space(nome).params:
            if p.name not in vistos:
                vistos.add(p.name)
                params.append(p)
    return SearchSpace(params)
