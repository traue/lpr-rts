"""Utilitarios geometricos puros (numpy): IoU, homografia, cruzamento de linhas.

Mantido sem dependencia de OpenCV para que possa ser testado isoladamente e
usado dentro do laco quente do rastreador sem custo de conversao.
"""

from __future__ import annotations

import math

import numpy as np

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union entre duas caixas ``(x1, y1, x2, y2)``."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU vetorizado. ``boxes_a`` (N,4) x ``boxes_b`` (M,4) -> (N,M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)


def contains(outer: BBox, inner: BBox, tol: float = 0.0) -> bool:
    """True se ``inner`` esta (quase) contida em ``outer``."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (ix1 >= ox1 - tol and iy1 >= oy1 - tol and ix2 <= ox2 + tol and iy2 <= oy2 + tol)


def containment_ratio(inner: BBox, outer: BBox) -> float:
    """Fracao da area de ``inner`` que cai dentro de ``outer`` (0..1)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    x1, y1 = max(ix1, ox1), max(iy1, oy1)
    x2, y2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / area) if area > 0 else 0.0


def clip_bbox(box: BBox, width: int, height: int) -> BBox:
    """Restringe a caixa aos limites da imagem."""
    x1, y1, x2, y2 = box
    x1 = float(min(max(0.0, x1), width - 1))
    y1 = float(min(max(0.0, y1), height - 1))
    x2 = float(min(max(0.0, x2), width))
    y2 = float(min(max(0.0, y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def expand_bbox(box: BBox, ratio: float, width: int, height: int) -> BBox:
    """Dilata a caixa por ``ratio`` (0.1 = +10% em cada lado) e recorta na imagem."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio, h * ratio
    return clip_bbox((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


# ---------------------------------------------------------------------------
# Linhas virtuais de contagem
# ---------------------------------------------------------------------------

def side_of_line(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Produto vetorial 2D: sinal indica de que lado da reta AB o ponto P esta."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    """True se o segmento P1P2 cruza o segmento Q1Q2 (casos colineares ignorados)."""
    d1 = side_of_line(p1, q1, q2)
    d2 = side_of_line(p2, q1, q2)
    d3 = side_of_line(q1, p1, p2)
    d4 = side_of_line(q2, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def crossing_direction(
    prev: tuple[float, float],
    curr: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> str:
    """Sentido do cruzamento da linha AB: ``"in"`` (neg->pos) ou ``"out"``."""
    return "in" if side_of_line(prev, a, b) < 0 <= side_of_line(curr, a, b) else "out"


# ---------------------------------------------------------------------------
# Homografia imagem -> plano do solo (para velocidade em km/h)
# ---------------------------------------------------------------------------

def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estima a homografia 3x3 que leva ``src`` (4x2, pixels) em ``dst`` (4x2, metros).

    Implementacao direta do DLT com SVD - evita depender de ``cv2`` aqui, o que
    mantem o modulo testavel sem OpenCV e o resultado identico a cv2.getPerspectiveTransform
    (a menos de escala, que normalizamos por H[2,2]).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4 or dst.shape[0] != src.shape[0]:
        raise ValueError("homografia exige >= 4 pares de pontos correspondentes")

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a_mat = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a_mat)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica a homografia a pontos (N,2), devolvendo coordenadas no plano destino."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts, ones]) @ h.T
    w = homo[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homo[:, :2] / w


def speed_kmh(
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    dt_s: float,
    h: np.ndarray | None = None,
    meters_per_pixel: float | None = None,
) -> float | None:
    """Velocidade escalar entre dois pontos de imagem.

    Usa a homografia quando disponivel (preciso); caso contrario, uma escala
    uniforme metros/pixel (aproximacao valida so para camera quase perpendicular).
    Retorna ``None`` se nao houver calibracao ou ``dt_s`` invalido.
    """
    if dt_s <= 1e-6:
        return None
    if h is not None:
        ground = apply_homography(h, np.array([p_from, p_to]))
        dist_m = float(np.linalg.norm(ground[1] - ground[0]))
    elif meters_per_pixel is not None:
        dist_m = math.dist(p_from, p_to) * meters_per_pixel
    else:
        return None
    return dist_m / dt_s * 3.6
