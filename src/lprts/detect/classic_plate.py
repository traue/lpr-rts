"""Detector de placas classico (morfologico) - baseline sem rede neural.

Serve a dois propositos:

1. **Executabilidade imediata**: o projeto roda de ponta a ponta sem baixar
   nenhum peso, o que facilita a primeira execucao e a instalacao em maquinas
   sem rede.
2. **Baseline cientifico**: da um ponto de comparacao classico (pre-deep-learning)
   contra o detector YOLO, algo util de reportar numa tese.

Ideia: caracteres de placa produzem uma concentracao densa de bordas verticais.
Um Sobel em X seguido de fechamento morfologico com nucleo largo funde esses
caracteres num unico blob retangular, filtrado depois por geometria.

Limitacoes (documentadas de proposito): sensivel a iluminacao, falha com placas
muito inclinadas ou pequenas (< 25 px de altura) e produz mais falsos positivos
que o YOLO. Nao e o modo recomendado para os resultados finais.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Detection
from ..utils.geometry import clip_bbox
from .base import Detector, nms

# Placa brasileira: 400 x 130 mm -> razao nominal ~3.08.
# A faixa aceita cobre perspectiva moderada e recorte imperfeito.
BR_ASPECT_MIN = 1.9
BR_ASPECT_MAX = 5.5


class ClassicPlateDetector(Detector):
    """Detecta regioes candidatas a placa por analise de bordas e morfologia."""

    name = "classic_plate"

    def __init__(
        self,
        min_height: int = 18,
        max_height_ratio: float = 0.5,
        aspect_min: float = BR_ASPECT_MIN,
        aspect_max: float = BR_ASPECT_MAX,
        min_extent: float = 0.35,
        max_candidates: int = 8,
        conf: float = 0.20,
    ) -> None:
        self.min_height = min_height
        self.max_height_ratio = max_height_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.min_extent = min_extent
        self.max_candidates = max_candidates
        self.conf = conf

    def describe(self) -> dict:
        return {
            "name": self.name, "runtime": "opencv", "device": "cpu",
            "params": {
                "min_height": self.min_height, "aspect": [self.aspect_min, self.aspect_max],
                "min_extent": self.min_extent, "conf": self.conf,
            },
        }

    # -- heuristica de pontuacao ------------------------------------------

    def _score(self, roi: np.ndarray, aspect: float, extent: float) -> float:
        """Pontua um candidato combinando geometria e densidade de caracteres.

        Sem rede neural nao existe "confianca" calibrada; produzimos um score em
        [0,1] comparavel entre candidatos, mas que **nao** deve ser interpretado
        como probabilidade.
        """
        if roi.size == 0:
            return 0.0
        # 1. Proximidade da razao de aspecto nominal da placa BR.
        aspect_score = max(0.0, 1.0 - abs(aspect - 3.08) / 2.5)
        # 2. Preenchimento do retangulo pelo contorno.
        extent_score = min(1.0, extent / 0.8)
        # 3. Densidade de componentes com forma de caractere.
        char_score = min(1.0, self._char_like_components(roi) / 7.0)
        # 4. Contraste local: placas sao alto-contraste por norma.
        contrast_score = min(1.0, float(roi.std()) / 60.0)
        return float(
            0.30 * aspect_score + 0.20 * extent_score + 0.35 * char_score + 0.15 * contrast_score
        )

    @staticmethod
    def _char_like_components(roi_gray: np.ndarray) -> int:
        """Conta componentes conexos com proporcao/altura tipicas de caractere."""
        if roi_gray.size == 0 or roi_gray.shape[0] < 8:
            return 0
        thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        h_roi = roi_gray.shape[0]
        count = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < 0.35 * h_roi or h > 0.95 * h_roi:
                continue
            if w == 0 or not (0.15 <= w / h <= 1.2):
                continue
            if area < 0.15 * w * h:
                continue
            count += 1
        return count

    # -- deteccao ----------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.bilateralFilter(gray, 5, 40, 40)

        # Bordas verticais: caracteres geram gradiente forte em X.
        sobel = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel = cv2.convertScaleAbs(sobel)
        binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Nucleo largo e baixo: funde caracteres vizinhos numa mancha de placa.
        kernel_w = max(9, w_img // 40 | 1)
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        )
        closed = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_h = max(self.min_height + 1, int(h_img * self.max_height_ratio))

        candidates: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < self.min_height or h > max_h or w < 2 * h * 0.5:
                continue
            aspect = w / float(h)
            if not (self.aspect_min <= aspect <= self.aspect_max):
                continue
            extent = cv2.contourArea(contour) / float(max(w * h, 1))
            if extent < self.min_extent:
                continue
            score = self._score(gray[y : y + h, x : x + w], aspect, extent)
            if score < self.conf:
                continue
            candidates.append(
                Detection(
                    bbox=clip_bbox((float(x), float(y), float(x + w), float(y + h)), w_img, h_img),
                    score=score, label="plate", class_id=0,
                )
            )

        candidates = nms(candidates, iou_threshold=0.3)
        return candidates[: self.max_candidates]
