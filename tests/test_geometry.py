"""Geometria: IoU, homografia, linhas virtuais."""

from __future__ import annotations

import numpy as np
import pytest

from lprts.utils import geometry as geo


class TestIoU:
    def test_identicas(self):
        assert geo.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjuntas(self):
        assert geo.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_sobreposicao_parcial(self):
        # Interseccao 5x10=50; uniao 100+100-50=150
        assert geo.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_matriz_bate_com_escalar(self):
        a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        m = geo.iou_matrix(a, b)
        assert m.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert m[i, j] == pytest.approx(geo.iou(tuple(a[i]), tuple(b[j])), abs=1e-6)

    def test_matriz_vazia(self):
        assert geo.iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


class TestContencao:
    def test_contida(self):
        assert geo.contains((0, 0, 100, 100), (10, 10, 50, 50))

    def test_razao_de_contencao(self):
        # Metade da caixa interna cai fora da externa
        assert geo.containment_ratio((50, 0, 150, 100), (0, 0, 100, 100)) == pytest.approx(0.5)

    def test_recorte_nos_limites(self):
        assert geo.clip_bbox((-10, -10, 200, 200), 100, 80) == (0.0, 0.0, 100.0, 80.0)

    def test_dilatacao(self):
        x1, y1, x2, y2 = geo.expand_bbox((40, 40, 60, 60), 0.5, 200, 200)
        assert (x1, y1, x2, y2) == (30.0, 30.0, 70.0, 70.0)


class TestHomografia:
    def test_mapeia_os_pontos_de_referencia(self):
        src = np.array([[400, 700], [880, 700], [760, 400], [520, 400]], dtype=float)
        dst = np.array([[0, 0], [3.5, 0], [3.5, 20.0], [0, 20.0]], dtype=float)
        h = geo.homography_from_points(src, dst)
        obtido = geo.apply_homography(h, src)
        assert np.allclose(obtido, dst, atol=1e-6)

    def test_exige_quatro_pontos(self):
        with pytest.raises(ValueError):
            geo.homography_from_points(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_velocidade_sem_calibracao_e_nula(self):
        assert geo.speed_kmh((0, 0), (10, 0), 1.0) is None

    def test_velocidade_com_escala_uniforme(self):
        # 10 px a 0,5 m/px em 1 s = 5 m/s = 18 km/h
        v = geo.speed_kmh((0, 0), (10, 0), 1.0, meters_per_pixel=0.5)
        assert v == pytest.approx(18.0)

    def test_dt_invalido(self):
        assert geo.speed_kmh((0, 0), (10, 0), 0.0, meters_per_pixel=0.5) is None


class TestLinhasVirtuais:
    def test_cruzamento_detectado(self):
        assert geo.segments_intersect((0, 0), (0, 10), (-5, 5), (5, 5))

    def test_sem_cruzamento(self):
        assert not geo.segments_intersect((0, 0), (0, 3), (-5, 5), (5, 5))

    def test_sentidos_opostos(self):
        a, b = (-5.0, 5.0), (5.0, 5.0)
        descendo = geo.crossing_direction((0, 0), (0, 10), a, b)
        subindo = geo.crossing_direction((0, 10), (0, 0), a, b)
        assert descendo != subindo


class TestRecorteGirado:
    """Video vertical de celular: o quadro chega deitado e so o recorte e girado."""

    @pytest.mark.parametrize("rotacao", [0, 1, 2])   # 90 horario, 180, 90 anti-horario
    @pytest.mark.parametrize("caixa,pad", [((10, 20, 30, 50), 0.0), ((3, 7, 40, 28), 0.0),
                                           ((10, 20, 30, 50), 0.1)])
    def test_igual_ao_recorte_da_imagem_girada(self, rotacao, caixa, pad):
        import cv2

        from lprts.utils.imgproc import crop, crop_rotated

        cru = np.random.default_rng(0).integers(0, 255, (60, 100, 3), dtype=np.uint8)
        girada = cv2.rotate(cru, rotacao)
        esperado = crop(girada, caixa, pad)
        assert np.array_equal(crop_rotated(cru, caixa, rotacao, pad), esperado)
