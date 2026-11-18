"""Inferencia para as metricas de avaliacao: intervalos de confianca e testes pareados.

Um numero sem incerteza nao sustenta comparacao: "93% contra 90%" num video com
40 veiculos pode ser so ruido. Aqui ficam as ferramentas minimas para reportar
corretamente:

* **Bootstrap por unidade** - reamostra as unidades independentes (veiculos,
  eventos) e devolve o intervalo percentil. Com semente fixa: reproduzivel.
* **Bootstrap por agrupamento** - para metricas sobre leituras, que NAO sao
  independentes: as leituras de um mesmo veiculo sao altamente correlacionadas
  (mesma placa, mesma iluminacao). Reamostrar leituras como se fossem
  independentes produziria intervalos estreitos demais; reamostramos veiculos.
* **McNemar exato** - para comparar duas configuracoes (ou fusao contra leitura
  unica) nos MESMOS veiculos: so os pares discordantes informam.

Sem dependencia de scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Media e IC percentil por bootstrap de unidades independentes.

    Devolve ``(estimativa, limite_inferior, limite_superior)``. Com menos de duas
    unidades o intervalo e degenerado (``nan``).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    est = float(arr.mean())
    if arr.size < 2:
        return (est, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medias = arr[idx].mean(axis=1)
    lo, hi = np.quantile(medias, [alpha / 2, 1 - alpha / 2])
    return (est, float(lo), float(hi))


def ratio_bootstrap_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Razao de somas com IC por bootstrap de agrupamentos.

    Cada posicao e um agrupamento (ex.: um veiculo) com ``numerador`` acertos em
    ``denominador`` tentativas. A estimativa e sum(num)/sum(den) - a mesma do
    calculo agregado - e o intervalo reamostra agrupamentos inteiros.
    """
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if num.size == 0 or den.sum() <= 0:
        return (float("nan"), float("nan"), float("nan"))
    est = float(num.sum() / den.sum())
    if num.size < 2:
        return (est, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    somas_den = den[idx].sum(axis=1)
    validos = somas_den > 0
    razoes = num[idx].sum(axis=1)[validos] / somas_den[validos]
    if razoes.size == 0:
        return (est, float("nan"), float("nan"))
    lo, hi = np.quantile(razoes, [alpha / 2, 1 - alpha / 2])
    return (est, float(lo), float(hi))


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporcao (bom mesmo com n pequeno)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def _binom_cdf(k: int, n: int) -> float:
    """P(X <= k) para X ~ Binomial(n, 1/2), exato."""
    return sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bilateral do teste de McNemar exato (binomial).

    ``b`` = pares em que so A acertou; ``c`` = pares em que so B acertou. Os
    pares concordantes nao entram: nao discriminam as duas configuracoes.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * _binom_cdf(k, n)))


def paired_comparison(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Compara dois resultados binarios nas MESMAS unidades (ex.: por veiculo).

    Devolve as taxas de acerto, a diferenca (B - A) com IC por bootstrap pareado,
    os pares discordantes e o p-valor de McNemar exato.
    """
    if len(a) != len(b):
        raise ValueError("as duas listas precisam ter o mesmo tamanho (mesmas unidades)")
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    so_a = int(np.sum((xa == 1) & (xb == 0)))
    so_b = int(np.sum((xa == 0) & (xb == 1)))
    diff, lo, hi = bootstrap_ci((xb - xa).tolist()) if xa.size else (float("nan"),) * 3
    return {
        "n": int(xa.size),
        "rate_a": float(xa.mean()) if xa.size else float("nan"),
        "rate_b": float(xb.mean()) if xb.size else float("nan"),
        "diff_b_minus_a": diff,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "only_a_correct": so_a,
        "only_b_correct": so_b,
        "mcnemar_p": mcnemar_exact(so_a, so_b),
    }
