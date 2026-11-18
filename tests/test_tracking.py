"""Rastreamento: Kalman, atribuicao hungara e ByteTrack."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lprts.core.types import Detection
from lprts.track import bytetrack
from lprts.track.assignment import linear_sum_assignment, max_weight_assignment
from lprts.track.bytetrack import ByteTracker
from lprts.track.kalman import KalmanBoxFilter, bbox_to_z, z_to_bbox


class TestKalman:
    def test_conversao_ida_e_volta(self):
        box = (100.0, 50.0, 180.0, 210.0)
        assert z_to_bbox(bbox_to_z(box)) == pytest.approx(box, abs=1e-6)

    def test_aprende_velocidade_constante(self):
        kf = KalmanBoxFilter((100, 100, 140, 180))
        for t in range(1, 16):
            kf.predict()
            kf.update((100 + 5 * t, 100, 140 + 5 * t, 180))
        vx, vy = kf.velocity
        assert vx == pytest.approx(5.0, abs=0.3)
        assert vy == pytest.approx(0.0, abs=0.3)

    def test_predicao_extrapola(self):
        kf = KalmanBoxFilter((0, 0, 40, 80))
        for t in range(1, 11):
            kf.predict()
            kf.update((10 * t, 0, 40 + 10 * t, 80))
        assert kf.predict()[0] == pytest.approx(110.0, abs=2.0)

    def test_covariancia_permanece_simetrica(self):
        """A forma de Joseph preserva a simetria de P ao longo do tempo."""
        kf = KalmanBoxFilter((0, 0, 40, 80))
        for _ in range(50):
            kf.predict()
            kf.update((1, 1, 41, 81))
        assert np.allclose(kf.P, kf.P.T, atol=1e-8)


class TestAtribuicao:
    def test_otima_contra_forca_bruta(self):
        rng = np.random.default_rng(0)
        for _ in range(120):
            n, m = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            cost = rng.random((n, m)) * 10
            rows, cols = linear_sum_assignment(cost)
            obtido = cost[rows, cols].sum()
            k = min(n, m)
            melhor = min(
                sum(cost[i, j] for i, j in zip(r, c, strict=True))
                for r in itertools.combinations(range(n), k)
                for c in itertools.permutations(range(m), k)
            )
            assert obtido == pytest.approx(melhor, abs=1e-9)

    def test_matriz_vazia(self):
        rows, cols = linear_sum_assignment(np.zeros((0, 0)))
        assert len(rows) == 0 and len(cols) == 0

    def test_retangular(self):
        rows, cols = linear_sum_assignment(np.array([[1.0, 5.0, 9.0]]))
        assert len(rows) == 1 and cols[0] == 0

    def test_vetorizado_igual_ao_laco_escalar(self):
        """Mesma atribuicao, bit a bit, inclusive nos empates: nenhuma trilha muda."""
        rng = np.random.default_rng(7)
        for k in range(400):
            n, m = int(rng.integers(1, 9)), int(rng.integers(1, 9))
            cost = (rng.integers(0, 4, (n, m)).astype(float) if k % 2
                    else 1 - rng.random((n, m)) * (rng.random((n, m)) < 0.3))
            esperado = _hungaro_escalar(cost)
            obtido = linear_sum_assignment(cost)
            assert np.array_equal(esperado[0], obtido[0])
            assert np.array_equal(esperado[1], obtido[1])


def _hungaro_escalar(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O laco escalar da 0.4 (Jonker-Volgenant), referencia da versao vetorizada."""
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    transposto = n_rows > n_cols
    if transposto:
        cost = cost.T
        n_rows, n_cols = cost.shape
    u, v = np.zeros(n_rows + 1), np.zeros(n_cols + 1)
    p, way = np.zeros(n_cols + 1, dtype=int), np.zeros(n_cols + 1, dtype=int)
    for i in range(1, n_rows + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n_cols + 1, np.inf), np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    linhas = [p[j] - 1 for j in range(1, n_cols + 1) if p[j] > 0]
    colunas = [j - 1 for j in range(1, n_cols + 1) if p[j] > 0]
    ordem = np.argsort(linhas)
    r, c = np.asarray(linhas, dtype=int)[ordem], np.asarray(colunas, dtype=int)[ordem]
    return (c, r) if transposto else (r, c)


def _peso_maximo_por_forca_bruta(s: np.ndarray, limiar: float) -> float:
    """Definicao do lapjv com cost_limit: max soma(s - limiar) sobre pares admissiveis."""
    n, m = s.shape
    melhor = 0.0
    for k in range(1, min(n, m) + 1):
        for linhas in itertools.combinations(range(n), k):
            for colunas in itertools.permutations(range(m), k):
                pares = list(zip(linhas, colunas, strict=True))
                if all(s[i, j] >= limiar for i, j in pares):
                    melhor = max(melhor, sum(s[i, j] - limiar for i, j in pares))
    return melhor


class TestPesoMaximo:
    """O criterio de associacao do ByteTrack de referencia (lapjv com cost_limit)."""

    def test_otimo_contra_forca_bruta(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            n, m = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            s = rng.random((n, m)) * (rng.random((n, m)) < 0.6)
            limiar = float(rng.choice([0.2, 0.3, 0.5]))
            rows, cols = max_weight_assignment(s, limiar)
            assert all(s[i, j] >= limiar for i, j in zip(rows, cols, strict=True))
            assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
            obtido = sum(s[i, j] - limiar for i, j in zip(rows, cols, strict=True))
            assert obtido == pytest.approx(_peso_maximo_por_forca_bruta(s, limiar), abs=1e-6)

    def test_igual_ao_lapjv_estendido_do_bytetrack(self):
        """Mesma escolha da matriz estendida do lap (cost_limit/2 para ficar sem par)."""
        scipy_opt = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(3)
        for _ in range(300):
            n, m = int(rng.integers(1, 7)), int(rng.integers(1, 7))
            iou = rng.random((n, m)) * (rng.random((n, m)) < 0.5)
            limite = 0.8                     # match_thresh do ByteTrack: IoU minimo 0,2
            estendida = np.full((n + m, n + m), limite / 2)
            estendida[n:, m:] = 0.0
            estendida[:n, :m] = 1.0 - iou
            r, c = scipy_opt.linear_sum_assignment(estendida)
            lap = {(i, j) for i, j in zip(r, c, strict=True)
                   if i < n and j < m and 1.0 - iou[i, j] < limite}
            rows, cols = max_weight_assignment(iou, 1.0 - limite)
            assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == lap

    def test_par_forte_nao_e_trocado_por_dois_fracos(self):
        # Maximizar o numero de pares (criterio do MOTA) daria (0,1) e (1,0):
        # as duas trilhas trocariam de objeto.
        s = np.array([[0.29, 0.31], [0.32, 0.95]])
        rows, cols = max_weight_assignment(s, 0.3)
        assert list(zip(rows, cols, strict=True)) == [(1, 1)]

    def test_par_no_limiar_e_admissivel(self):
        rows, cols = max_weight_assignment(np.array([[0.5, 0.0]]), 0.5)
        assert list(zip(rows, cols, strict=True)) == [(0, 0)]

    def test_inadmissiveis_nunca_entram(self):
        s = np.array([[np.nan, 0.5, 0.1], [-np.inf, 0.4, np.inf]])
        rows, cols = max_weight_assignment(s, 0.3)
        assert list(zip(rows, cols, strict=True)) == [(0, 1)]
        assert max_weight_assignment(np.zeros((3, 0)), 0.3)[0].size == 0
        assert max_weight_assignment(np.full((2, 2), 0.1), 0.3)[0].size == 0


class TestByteTracker:
    def test_associacao_usa_o_peso_maximo(self, monkeypatch):
        """As duas trilhas mantem seus objetos em vez de trocar por pares mais fracos."""
        tracker = ByteTracker(min_hits=1)
        tracker.update([Detection((0, 0, 50, 50), 0.9, "car"),
                        Detection((100, 0, 150, 50), 0.9, "car")], 0)
        ious = np.array([[0.29, 0.31], [0.32, 0.95]], dtype=np.float32)
        monkeypatch.setattr(bytetrack, "iou_matrix", lambda a, b: ious)
        dets = [Detection((0, 0, 1, 1), 0.9, "car"), Detection((0, 0, 1, 1), 0.9, "car")]
        pares, livres_t, livres_d = tracker._associate(dets, [0, 1], 0.3)
        assert pares == [(1, 1)] and livres_t == [0] and livres_d == [0]

    def test_mantem_id_em_trajetoria_reta(self):
        tracker = ByteTracker(min_hits=3)
        ids = set()
        for f in range(20):
            saida = tracker.update([Detection((10 + f * 6, 100, 90 + f * 6, 180), 0.9, "car")], f)
            ids.update(t.track_id for t in saida)
        assert len(ids) == 1

    def test_nao_troca_id_em_cruzamento(self):
        """Dois veiculos em rotas cruzadas devem manter identidades distintas."""
        tracker = ByteTracker(min_hits=3)
        vistos = []
        for f in range(30):
            a = Detection((10 + f * 8, 100, 90 + f * 8, 180), 0.9, "car")
            b = Detection((250 - f * 8, 100, 330 - f * 8, 180), 0.85, "car")
            vistos.append({t.track_id for t in tracker.update([a, b], f)})
        todos = set().union(*vistos)
        assert len(todos) == 2

    def test_deteccao_fraca_nao_cria_trilha(self):
        tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=1)
        assert tracker.update([Detection((0, 0, 50, 50), 0.2, "car")], 0) == []

    def test_deteccao_fraca_sustenta_trilha_existente(self):
        """E a ideia central do ByteTrack: objeto ocluido nao pode perder o ID."""
        tracker = ByteTracker(min_hits=2, second_match_iou=0.3)
        for f in range(5):
            tracker.update([Detection((10 + f * 4, 10, 90 + f * 4, 90), 0.9, "car")], f)
        antes = {t.track_id for t in tracker.tracks}
        for f in range(5, 9):   # confianca cai (oclusao parcial)
            tracker.update([Detection((10 + f * 4, 10, 90 + f * 4, 90), 0.25, "car")], f)
        assert {t.track_id for t in tracker.tracks} == antes

    def test_trilha_expira(self):
        tracker = ByteTracker(max_age=3, min_hits=1)
        tracker.update([Detection((0, 0, 50, 50), 0.9, "car")], 0)
        for f in range(1, 8):
            tracker.update([], f)
        assert tracker.tracks == []

    def test_historico_da_trilha_e_limitado(self):
        """Execucao 24/7 nao pode crescer sem limite em memoria..."""
        from lprts.core.types import MAX_TRAJECTORY_POINTS

        tracker = ByteTracker(min_hits=1)
        n = MAX_TRAJECTORY_POINTS * 2 + 500
        for f in range(n):
            tracker.update([Detection((10, 10, 90, 90), 0.9, "car")], f, f / 30)
        t = tracker.tracks[0]
        assert len(t.centroids) <= MAX_TRAJECTORY_POINTS
        # ...mas a dizimacao nao pode cortar o comeco: permanencia e trajetoria
        # dependem do primeiro ponto (antes os 300 mais antigos eram apagados).
        assert t.frames[0] == 0 and t.frames[-1] == n - 1
        assert t.hits == n                     # n_frames gravado vem de hits, nao do historico
        assert len(t.frames) == len(t.timestamps) == len(t.ground_points)

    def test_classe_do_objeto_preservada(self):
        """Rotulos COCO em ingles sao traduzidos para a taxonomia interna."""
        tracker = ByteTracker(min_hits=1)
        saida = tracker.update([Detection((0, 0, 50, 50), 0.9, "truck")], 0)
        assert saida[0].object_class.value == "caminhao"
        assert saida[0].category.value == "veiculo_motorizado"
        assert saida[0].can_have_plate

    def test_classes_sem_placa_marcadas(self):
        """Pedestre e bicicleta nunca devem chegar ao OCR."""
        tracker = ByteTracker(min_hits=1)
        for rotulo, esperado in (("person", "pessoa"), ("bicycle", "bicicleta")):
            t = tracker.update([Detection((0, 0, 50, 50), 0.9, rotulo)], 0)[0]
            assert t.object_class.value == esperado
            assert not t.can_have_plate
            assert t.is_vulnerable
            tracker.reset()
