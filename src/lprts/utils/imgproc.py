"""Operacoes de imagem reutilizadas por OCR, UI e exportacao de recortes."""

from __future__ import annotations

import cv2
import numpy as np

BBox = tuple[float, float, float, float]


def crop(image: np.ndarray, box: BBox, pad: float = 0.0) -> np.ndarray:
    """Recorta ``box`` da imagem com ``pad`` proporcional; devolve copia contigua."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if pad > 0:
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = x1 - bw * pad, y1 - bh * pad
        x2, y2 = x2 + bw * pad, y2 + bh * pad
    xi1 = max(0, int(round(x1)))
    yi1 = max(0, int(round(y1)))
    xi2 = min(w, int(round(x2)))
    yi2 = min(h, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((0, 0, 3) if image.ndim == 3 else (0, 0), dtype=image.dtype)
    return np.ascontiguousarray(image[yi1:yi2, xi1:xi2])


def crop_rotated(raw: np.ndarray, box: BBox, rotation: int, pad: float = 0.0) -> np.ndarray:
    """``crop(cv2.rotate(raw, rotation), box, pad)`` sem girar a imagem inteira.

    ``box`` esta nas coordenadas da imagem GIRADA; recorta a regiao equivalente da
    imagem crua e gira so o recorte. Usado no video vertical de celular, cujo quadro
    4K chega deitado: girar o quadro inteiro custava 5,6 ms por quadro.
    """
    x1, y1, x2, y2 = box
    if pad > 0:
        bw, bh = x2 - x1, y2 - y1
        x1, y1, x2, y2 = x1 - bw * pad, y1 - bh * pad, x2 + bw * pad, y2 + bh * pad
    h, w = raw.shape[:2]
    if rotation == cv2.ROTATE_90_CLOCKWISE:        # girada (x', y') <- crua (y', h - x')
        cru = (y1, h - x2, y2, h - x1)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:  # girada (x', y') <- crua (w - y', x')
        cru = (w - y2, x1, w - y1, x2)
    elif rotation == cv2.ROTATE_180:
        cru = (w - x2, h - y2, w - x1, h - y1)
    else:
        raise ValueError(f"Rotacao desconhecida: {rotation!r}")
    patch = crop(raw, cru)
    return np.ascontiguousarray(cv2.rotate(patch, rotation)) if patch.size else patch


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpness(image: np.ndarray) -> float:
    """Variancia do Laplaciano - proxy classico de nitidez/foco.

    Usado para escolher o melhor quadro de uma trilha antes de rodar o OCR caro.
    """
    if image.size == 0:
        return 0.0
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(image: np.ndarray) -> tuple[float, float]:
    """Media e desvio-padrao da luminancia (0-255) - metadados de qualidade."""
    if image.size == 0:
        return 0.0, 0.0
    gray = to_gray(image)
    return float(gray.mean()), float(gray.std())


def letterbox(
    image: np.ndarray, size: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona preservando o aspecto e preenche com barras (padrao YOLO).

    Retorna ``(imagem, escala, (pad_x, pad_y))`` para desfazer a transformacao
    nas caixas previstas.
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=image.dtype)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def unletterbox_boxes(
    boxes: np.ndarray, scale: float, pad: tuple[int, int]
) -> np.ndarray:
    """Converte caixas do espaco letterbox de volta para a imagem original."""
    if len(boxes) == 0:
        return boxes
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    return out


def enhance_plate(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Normalizacao fotometrica de um recorte de placa antes do OCR.

    1. converte para escala de cinza;
    2. reamostra para ``target_height`` (LANCZOS4 amplia sem serrilhar);
    3. CLAHE para equalizar iluminacao local (sombra/sol forte);
    4. filtro bilateral, que suaviza ruido preservando as bordas dos caracteres.
    """
    if image.size == 0:
        return image
    gray = to_gray(image)
    h, w = gray.shape[:2]
    if h < 1 or w < 1:
        return gray
    scale = target_height / float(h)
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    gray = cv2.resize(gray, (max(1, int(round(w * scale))), target_height), interpolation=interp)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Corrige a inclinacao (roll) do recorte usando o retangulo minimo do texto.

    Rotacoes maiores que ``max_angle`` sao ignoradas: quase sempre indicam
    segmentacao ruim, e girar demais degrada mais do que ajuda.
    """
    if image.size == 0:
        return image
    gray = to_gray(image)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thr)
    if coords is None or len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Retifica um quadrilatero (placa em perspectiva) para um retangulo frontal."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )
    (tl, tr, br, bl) = ordered
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    mat = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, mat, (width, height))
