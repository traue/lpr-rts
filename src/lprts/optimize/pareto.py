"""Otimizacao multiobjetivo: dominancia de Pareto, NSGA-II e hipervolume.

## Por que multiobjetivo

Ajustar o pipeline nao tem resposta unica. Baixar ``ocr_every_n`` aumenta a
exatidao e derruba o FPS; subir ``min_reads`` aumenta a precisao e reduz a
revocacao. Reduzir tudo a um numero so - "acuracia menos 0,01 x latencia" -
esconde o compromisso atras de um peso arbitrario que ninguem sabe justificar.

A resposta honesta e a **fronteira de Pareto**: o conjunto de configuracoes em
que nao da para melhorar um objetivo sem piorar outro. O pesquisador escolhe o
ponto da fronteira segundo a aplicacao - um portico de pedagio tolera 10 fps se
a exatidao for alta; um sistema de contagem em tempo real, nao.

Implementacao propria em numpy: ordenacao por nao-dominancia e distancia de
aglomeracao do NSGA-II (Deb et al., 2002), mais hipervolume para comparar
fronteiras entre si.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Direction = Literal["max", "min"]


@dataclass(slots=True)
class Trial:
    """Uma configuracao avaliada e seus objetivos."""

    params: dict[str, object]
    objectives: dict[str, float]
    number: int = -1
    extra: dict[str, object] = field(default_factory=dict)
    rank: int = -1              # frente de Pareto (0 = melhor)
    crowding: float = 0.0       # distancia de aglomeracao dentro da frente

    def values(self, names: list[str]) -> list[float]:
        return [float(self.objectives.get(n, 0.0)) for n in names]

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "params": dict(self.params),
            "objectives": dict(self.objectives),
            "rank": self.rank,
            "crowding": self.crowding,
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


def _to_minimization(
    trials: list[Trial], names: list[str], directions: list[Direction]
) -> np.ndarray:
    """Matriz (n_trials, n_objetivos) convertida para minimizacao.

    Trabalhar sempre em minimizacao evita espalhar condicionais de sinal por
    toda a logica de dominancia.
    """
    matriz = np.array([t.values(names) for t in trials], dtype=np.float64)
    if matriz.size == 0:
        return matriz
    sinais = np.array([-1.0 if d == "max" else 1.0 for d in directions])
    return matriz * sinais


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """``a`` domina ``b`` (minimizacao): nao e pior em nada e melhor em algo."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(
    trials: list[Trial], names: list[str], directions: list[Direction]
) -> list[list[Trial]]:
    """Separa os trials em frentes de Pareto sucessivas (NSGA-II, secao III-A)."""
    if not trials:
        return []
    m = _to_minimization(trials, names, directions)
    n = len(trials)

    dominados_por: list[list[int]] = [[] for _ in range(n)]
    contagem = np.zeros(n, dtype=int)
    frentes: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(m[i], m[j]):
                dominados_por[i].append(j)
                contagem[j] += 1
            elif dominates(m[j], m[i]):
                dominados_por[j].append(i)
                contagem[i] += 1
        if contagem[i] == 0:
            frentes[0].append(i)

    # A contagem de i so fica definitiva apos comparar com todos; recolhe a
    # primeira frente novamente para incluir quem zerou depois do proprio laco.
    frentes[0] = [i for i in range(n) if contagem[i] == 0]

    atual = 0
    while frentes[atual]:
        proxima: list[int] = []
        for i in frentes[atual]:
            for j in dominados_por[i]:
                contagem[j] -= 1
                if contagem[j] == 0:
                    proxima.append(j)
        atual += 1
        frentes.append(proxima)
    frentes.pop()

    resultado: list[list[Trial]] = []
    for rank, frente in enumerate(frentes):
        grupo = []
        for i in frente:
            trials[i].rank = rank
            grupo.append(trials[i])
        resultado.append(grupo)
    return resultado


def crowding_distance(
    trials: list[Trial], names: list[str], directions: list[Direction]
) -> None:
    """Distancia de aglomeracao: preserva a diversidade ao longo da frente.

    Sem ela, a busca converge para um aglomerado num trecho da fronteira e o
    pesquisador perde justamente as opcoes dos extremos - por exemplo, a
    configuracao mais rapida que ainda e aceitavel.
    """
    n = len(trials)
    if n == 0:
        return
    if n <= 2:
        for t in trials:
            t.crowding = float("inf")
        return

    for t in trials:
        t.crowding = 0.0
    m = _to_minimization(trials, names, directions)

    for k in range(m.shape[1]):
        ordem = np.argsort(m[:, k])
        trials[ordem[0]].crowding = float("inf")
        trials[ordem[-1]].crowding = float("inf")
        faixa = m[ordem[-1], k] - m[ordem[0], k]
        if faixa < 1e-12:
            continue
        for pos in range(1, n - 1):
            i = ordem[pos]
            if trials[i].crowding == float("inf"):
                continue
            trials[i].crowding += float(
                (m[ordem[pos + 1], k] - m[ordem[pos - 1], k]) / faixa
            )


def pareto_front(
    trials: list[Trial], names: list[str], directions: list[Direction]
) -> list[Trial]:
    """Somente a primeira frente, ordenada pelo primeiro objetivo."""
    frentes = non_dominated_sort(trials, names, directions)
    if not frentes:
        return []
    frente = frentes[0]
    crowding_distance(frente, names, directions)
    reverso = directions[0] == "max"
    return sorted(frente, key=lambda t: t.objectives.get(names[0], 0.0), reverse=reverso)


def nsga2_survivors(
    trials: list[Trial], names: list[str], directions: list[Direction], n_keep: int
) -> list[Trial]:
    """Seleciona ``n_keep`` sobreviventes por rank e, no empate, por aglomeracao."""
    sobreviventes: list[Trial] = []
    for frente in non_dominated_sort(trials, names, directions):
        crowding_distance(frente, names, directions)
        if len(sobreviventes) + len(frente) <= n_keep:
            sobreviventes.extend(frente)
        else:
            faltam = n_keep - len(sobreviventes)
            frente.sort(key=lambda t: t.crowding, reverse=True)
            sobreviventes.extend(frente[:faltam])
            break
    return sobreviventes


# Pior valor possivel dos objetivos limitados que este projeto otimiza. Servem de
# ponto de referencia FIXO do hipervolume.
_PIOR_VALOR = {
    "precisao": 0.0, "revocacao": 0.0, "f1": 0.0, "taxa_publicacao": 0.0,
    "exatidao": 0.0, "cobertura": 0.0, "cer": 1.0,
}


def default_reference(names: list[str], directions: list[Direction]) -> list[float] | None:
    """Ponto de referencia fixo (no espaco ORIGINAL dos objetivos), se todos forem limitados.

    Um ponto de referencia derivado da propria fronteira muda de estudo para
    estudo e torna os hipervolumes incomparaveis - justamente o uso que justifica
    calcula-los. Para objetivos sem limite conhecido devolve ``None``.
    """
    if not all(n in _PIOR_VALOR for n in names):
        return None
    return [_PIOR_VALOR[n] for n in names]


def hypervolume(
    trials: list[Trial],
    names: list[str],
    directions: list[Direction],
    reference: list[float] | None = None,
) -> float:
    """Hipervolume dominado pela fronteira, em relacao a um ponto de referencia.

    E a metrica padrao para comparar duas fronteiras com um unico numero: quanto
    maior, melhor a fronteira em cobertura e proximidade do otimo. Implementado
    por varredura exata em 2 objetivos e por Monte Carlo acima disso.

    ``reference`` esta no espaco ORIGINAL dos objetivos (ex.: ``[0, 0]`` para
    precisao e revocacao). Sem ele, usa-se o ponto fixo de ``default_reference``;
    so na falta deste (objetivos sem limite conhecido) o ponto e derivado da
    fronteira - e o hipervolume deixa de ser comparavel entre estudos.
    """
    frente = pareto_front(trials, names, directions)
    if not frente:
        return 0.0
    m = _to_minimization(frente, names, directions)
    sinais = np.array([-1.0 if d == "max" else 1.0 for d in directions])
    if reference is None:
        reference = default_reference(names, directions)
    ref = (
        np.array(reference, dtype=np.float64) * sinais
        if reference is not None
        else m.max(axis=0) + np.abs(m.max(axis=0)) * 0.1 + 1e-9
    )

    if m.shape[1] == 1:
        return float(max(0.0, ref[0] - m[:, 0].min()))

    if m.shape[1] == 2:
        ordem = np.argsort(m[:, 0])
        pontos = m[ordem]
        volume = 0.0
        y_anterior = ref[1]
        for x, y in pontos:
            if x >= ref[0] or y >= y_anterior:
                continue
            volume += (ref[0] - x) * (y_anterior - y)
            y_anterior = y
        return float(volume)

    # >2 objetivos: estimativa por Monte Carlo dentro da caixa [min, ref].
    rng = np.random.default_rng(0)
    baixo = m.min(axis=0)
    caixa = float(np.prod(np.maximum(ref - baixo, 0.0)))
    if caixa <= 0:
        return 0.0
    amostras = rng.uniform(baixo, ref, size=(20000, m.shape[1]))
    dominados = np.any(np.all(m[None, :, :] <= amostras[:, None, :], axis=2), axis=1)
    return float(caixa * dominados.mean())
