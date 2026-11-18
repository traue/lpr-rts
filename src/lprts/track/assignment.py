"""Algoritmo Hungaro (Kuhn-Munkres) em numpy puro.

Resolve a associacao otima deteccao<->trilha minimizando o custo total. Uma
associacao gulosa e mais simples, mas troca identidades quando dois veiculos se
cruzam - exatamente o caso que estraga a re-identificacao de placas. Como as
matrizes aqui sao pequenas (dezenas de objetos), o O(n^3) e irrelevante e
evitamos depender de ``scipy`` so por esta funcao.
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna ``(linhas, colunas)`` da atribuicao de custo minimo.

    Equivalente a ``scipy.optimize.linear_sum_assignment`` para matrizes finitas.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("cost deve ser uma matriz bidimensional finita")
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n_rows, n_cols = cost.shape
    transposed = n_rows > n_cols
    if transposed:
        cost = cost.T
        n_rows, n_cols = cost.shape

    # Variante de Jonker-Volgenant com potenciais (caminho de aumento por linha).
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=int)   # p[j] = linha atribuida a coluna j
    way = np.zeros(n_cols + 1, dtype=int)

    # Os dois lacos por coluna rodam vetorizados: o rastreador resolve isto duas
    # vezes por quadro, na etapa serial do pipeline. Mesmas operacoes na mesma
    # ordem e o mesmo desempate (a primeira coluna de menor folga, como o
    # argmin): o resultado e bit a bit o do laco escalar.
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            livres = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            melhora = livres & (cur < minv[1:])
            minv[1:][melhora] = cur[melhora]
            way[1:][melhora] = j0
            folga = np.where(livres, minv[1:], np.inf)
            j1 = int(np.argmin(folga)) + 1
            delta = folga[j1 - 1]
            # Linhas das colunas usadas sao distintas (um emparelhamento), entao
            # a soma por indice equivale ao laco.
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows, cols = [], []
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    order = np.argsort(rows)
    rows_arr = np.asarray(rows, dtype=int)[order]
    cols_arr = np.asarray(cols, dtype=int)[order]
    if transposed:
        return cols_arr, rows_arr
    return rows_arr, cols_arr


def gated_assignment(
    similarities: np.ndarray, min_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximiza pares admissiveis, depois sua similaridade total (avaliacao CLEAR-MOT).

    E o criterio do py-motmetrics para o MOTA: o maior NUMERO de pares acima do
    limiar e, entre esses, a maior similaridade. Serve a metrica, nao ao
    rastreador - la ele trocaria um par forte por dois fracos (ver
    :func:`max_weight_assignment`).

    Colunas ficticias permitem deixar linhas sem par. A penalidade de um par
    ausente excede a variacao total dos custos validos; assim uma associacao
    invalida nunca rouba uma alternativa valida, mesmo em matrizes retangulares.
    NaN e infinito sao inadmissiveis.
    """
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(min_similarity):
        raise ValueError("similarities deve ser 2D e min_similarity finito")
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    valid = np.isfinite(values) & (values >= min_similarity)
    n_rows, n_cols = values.shape
    penalty = float(min(n_rows, n_cols) + 1)
    costs = np.full((n_rows, n_cols + n_rows), penalty)
    costs[:, :n_cols] = 2 * penalty
    if valid.any():
        usable = values[valid]
        lo, hi = float(np.min(usable)), float(np.max(usable))
        costs[:, :n_cols][valid] = (hi - usable) / (hi - lo) if hi > lo else 0.0
    rows, cols = linear_sum_assignment(costs)
    real = cols < n_cols
    return rows[real], cols[real]


# Desempate a favor de casar um par exatamente no limiar (ganho zero). Fica
# abaixo da resolucao do IoU calculado em float32 (~1e-7), entao nao decide
# nenhuma escolha entre pares de similaridades realmente diferentes.
_EMPATE = 1e-9


def max_weight_assignment(
    similarities: np.ndarray, min_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Emparelhamento de peso maximo com limiar: maximiza a soma de ``s - limiar``.

    E o criterio do ByteTrack de referencia, ``lap.lapjv(1 - IoU,
    extend_cost=True, cost_limit=1 - limiar)``: linhas e colunas podem ficar
    sem par, e um par so entra se ganha mais do que deixar os dois livres. Ao
    contrario de :func:`gated_assignment`, nao maximiza o NUMERO de pares: com
    limiar 0,3, a trilha que cobre uma deteccao com IoU 0,95 fica com ela,
    em vez de as duas trilhas trocarem de objeto por dois pares de 0,31 e 0,32.

    Pares no limiar (``s == limiar``) sao admissiveis; NaN e infinito nao.
    Linhas e colunas sem nenhum par admissivel ficam de fora da conta (sairiam
    sem par de qualquer jeito), o que encolhe as matrizes esparsas de cena.
    """
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(min_similarity):
        raise ValueError("similarities deve ser 2D e min_similarity finito")
    vazio = (np.array([], dtype=int), np.array([], dtype=int))
    valid = np.isfinite(values) & (values >= min_similarity)
    if not valid.any():
        return vazio
    linhas = np.flatnonzero(valid.any(axis=1))
    colunas = np.flatnonzero(valid.any(axis=0))
    sub_validos = valid[np.ix_(linhas, colunas)]
    ganho = values[np.ix_(linhas, colunas)] - min_similarity + _EMPATE
    n_rows, n_cols = sub_validos.shape
    # Colunas extras: cada linha pode ficar sem par a custo 0. Um par inadmissivel
    # custa mais que isso e nunca entra; um admissivel custa -ganho.
    custos = np.zeros((n_rows, n_cols + n_rows))
    custos[:, :n_cols] = np.where(sub_validos, -ganho, 1.0)
    rows, cols = linear_sum_assignment(custos)
    reais = cols < n_cols
    rows, cols = rows[reais], cols[reais]
    admissiveis = sub_validos[rows, cols]
    return linhas[rows[admissiveis]], colunas[cols[admissiveis]]
