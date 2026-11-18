"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def plate_image():
    """Fabrica de imagens sinteticas de placa (fundo claro, texto escuro)."""

    def _make(text: str = "ABC1D23", width: int = 300, height: int = 100, blur: int = 0):
        img = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), (30, 30, 30), 3)

        # A escala da fonte e derivada do tamanho pedido para que o texto sempre
        # caiba com ~8% de margem lateral. Com escala fixa, um recorte estreito
        # cortaria o ultimo caractere e o teste mediria o fixture, nao o codigo.
        font = cv2.FONT_HERSHEY_SIMPLEX
        alvo = width * 0.84
        escala = 1.0
        (tw, _), _ = cv2.getTextSize(text, font, escala, 2)
        escala = max(0.4, alvo / max(tw, 1))
        espessura = max(2, int(round(escala * 2.6)))
        (tw, th), _ = cv2.getTextSize(text, font, escala, espessura)
        org = ((width - tw) // 2, (height + th) // 2)
        cv2.putText(img, text, org, font, escala, (20, 20, 20), espessura, cv2.LINE_AA)

        if blur >= 3:
            img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
        return img

    return _make


@pytest.fixture
def scene_with_plate(plate_image):
    """Cena maior com uma placa colada numa posicao conhecida."""

    def _make(text: str = "ABC1D23", box=(400, 300, 620, 373)):
        scene = np.full((720, 1280, 3), 95, dtype=np.uint8)
        x1, y1, x2, y2 = box
        plate = cv2.resize(plate_image(text), (x2 - x1, y2 - y1))
        scene[y1:y2, x1:x2] = plate
        return scene, (float(x1), float(y1), float(x2), float(y2))

    return _make


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "teste.db"
