"""Filtro de Kalman de velocidade constante para caixas delimitadoras.

Estado de 8 dimensoes ``[cx, cy, a, h, vcx, vcy, va, vh]``, onde ``a`` e a razao
largura/altura e ``h`` a altura. E a formulacao classica de SORT/DeepSORT:
prever a caixa no proximo quadro estabiliza a associacao quando o detector
falha por um ou dois quadros (oclusao parcial, desfoque de movimento).

Implementado em numpy puro - a matriz e 8x8, entao o custo e desprezivel e
evita-se uma dependencia so por isto.

O passo de predicao aceita ``dt`` em quadros da fonte. Quando quadros sao
descartados (tempo real) ou a deteccao e pulada (``detect_every_n``), entre duas
atualizacoes passam varios quadros; prever um unico passo faria a caixa prevista
ficar para tras de um veiculo rapido e a associacao falhar.
"""

from __future__ import annotations

import numpy as np

BBox = tuple[float, float, float, float]


def bbox_to_z(bbox: BBox) -> np.ndarray:
    """``(x1,y1,x2,y2)`` -> medida ``[cx, cy, a, h]``."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> BBox:
    """``[cx, cy, a, h]`` -> ``(x1,y1,x2,y2)``."""
    cx, cy, a, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    h = max(h, 1e-6)
    w = max(a * h, 1e-6)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxFilter:
    """Filtro de Kalman linear para uma unica caixa rastreada."""

    def __init__(self, bbox: BBox, dt: float = 1.0) -> None:
        self.dt = dt

        # Transicao: posicao integra velocidade; velocidade e constante.
        self.F = self._transition(dt)

        # Observamos apenas a caixa, nunca a velocidade.
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # Ruido de medida: altura e centro sao confiaveis; razao de aspecto oscila.
        self.R = np.diag([1.0, 1.0, 10.0, 1.0])
        # Ruido de processo: velocidades sao muito mais incertas que posicoes.
        self.Q = np.diag([1.0, 1.0, 0.01, 1.0, 0.01, 0.01, 1e-4, 0.01])

        # Covariancia inicial: alta nas velocidades, que sao desconhecidas.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4, 1e4])

        # Anotado como ndarray generico: com o numpy 2.2 (o do Python 3.10), np.zeros(8)
        # tem forma tipada (tuple[int]) e o mypy recusa os produtos matriciais abaixo.
        self.x: np.ndarray = np.zeros(8, dtype=np.float64)
        self.x[:4] = bbox_to_z(bbox)

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        f = np.eye(8, dtype=np.float64)
        for i in range(4):
            f[i, i + 4] = dt
        return f

    def predict(self, dt: float | None = None) -> BBox:
        """Avanca o estado ``dt`` quadros (padrao: o passo nominal) e devolve a caixa prevista."""
        passo = self.dt if dt is None else max(float(dt), 1e-6)
        f = self.F if passo == self.dt else self._transition(passo)
        self.x = f @ self.x
        # O ruido de processo acumula com o numero de passos (passeio aleatorio).
        self.P = f @ self.P @ f.T + self.Q * (passo / self.dt)
        # Altura e razao de aspecto negativas nao tem sentido fisico.
        self.x[2] = max(self.x[2], 1e-3)
        self.x[3] = max(self.x[3], 1e-3)
        return z_to_bbox(self.x[:4])

    def update(self, bbox: BBox) -> BBox:
        """Corrige o estado com uma nova medida."""
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        identity = np.eye(8, dtype=np.float64)
        # Forma de Joseph: preserva a simetria/positividade de P ao longo do tempo.
        factor = identity - K @ self.H
        self.P = factor @ self.P @ factor.T + K @ self.R @ K.T
        return z_to_bbox(self.x[:4])

    @property
    def bbox(self) -> BBox:
        return z_to_bbox(self.x[:4])

    @property
    def velocity(self) -> tuple[float, float]:
        """Velocidade do centro em pixels por quadro."""
        return float(self.x[4]), float(self.x[5])
