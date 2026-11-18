"""Desenho das anotacoes sobre o quadro (OpenCV puro, sem Qt).

Compartilhado pela interface grafica, pela gravacao de video anotado e pela
janela simples de fallback - assim o que se ve na tela e exatamente o que e
gravado em disco.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import UIConfig
from ..core.taxonomy import ObjectClass, label_pt
from ..core.types import FrameResult, TrackState

# Paleta em BGR (OpenCV). Cores escolhidas para contrastar com asfalto e ceu.
COLOR_VEHICLE = (120, 200, 90)
COLOR_PLATE = (60, 180, 255)      # provisorio / caixa de deteccao
COLOR_PLATE_OK = (80, 255, 120)   # leitura confirmada pela fusao
COLOR_TRACK = (230, 190, 90)
COLOR_LINE = (200, 120, 255)
COLOR_HUD_BG = (32, 30, 28)
COLOR_TEXT = (245, 245, 245)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Texto com tarja de fundo - legivel sobre qualquer cena."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(image, (x, y - th - baseline - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline + 1), FONT, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_track(image: np.ndarray, track: TrackState, cfg: UIConfig) -> None:
    """Desenha a caixa da trilha, o identificador e a placa consolidada."""
    x1, y1, x2, y2 = (int(v) for v in track.bbox)
    resolved = bool(track.plate_text)
    confirmed = resolved and track.plate_stable
    color = COLOR_PLATE_OK if confirmed else (COLOR_PLATE if resolved else COLOR_TRACK)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    parts = [f"#{track.track_id}"]
    if track.object_class not in (ObjectClass.DESCONHECIDO, ObjectClass.PLACA_VEICULAR):
        parts.append(label_pt(track.object_class))
    if track.speed_kmh is not None:
        parts.append(f"{track.speed_kmh:.0f} km/h")
    _label(image, " ".join(parts), (x1, y1 - 4), color, 0.45)

    if resolved:
        # O sufixo "?" e a cor distinta marcam leitura ainda nao consolidada.
        suffix = "" if confirmed else " ?"
        text = f"{track.plate_text}  {track.plate_confidence:.2f}{suffix}"
        _label(image, text, (x1, y2 + 20), color, 0.62, 2)

    if cfg.show_trajectories and len(track.centroids) > 1:
        pts = np.array(track.centroids[-40:], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], False, color, 1, cv2.LINE_AA)


def draw_hud(image: np.ndarray, result: FrameResult, metrics: dict | None = None) -> None:
    """Painel de telemetria no canto superior esquerdo."""
    t = result.timing
    lines = [
        f"quadro {result.frame.index}",
        f"latencia {t.latency_ms:5.1f} ms  servico {t.total_ms:.1f}  (det {t.detect_ms:.1f})",
        f"trilhas {len(result.tracks)}  placas {len(result.plates)}",
    ]
    if metrics:
        lines.insert(
            1, f"{metrics.get('fps', 0.0):5.1f} FPS   latencia p95 {metrics.get('p95', 0.0):.1f} ms"
        )
        lines.append(
            f"placas unicas {metrics.get('unique', 0)}   descartados {metrics.get('dropped', 0)}"
        )

    pad = 8
    widths = [cv2.getTextSize(t_, FONT, 0.5, 1)[0][0] for t_ in lines]
    box_w = max(widths) + pad * 2
    box_h = len(lines) * 20 + pad * 2

    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), COLOR_HUD_BG, -1)
    # Mistura em vez de opaco: nao esconder a cena por tras do painel.
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    cv2.rectangle(image, (10, 10), (10 + box_w, 10 + box_h), (90, 85, 80), 1)

    for i, text in enumerate(lines):
        cv2.putText(
            image, text, (10 + pad, 10 + pad + 14 + i * 20), FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA
        )


def draw_lines(image: np.ndarray, lines: list, scale: float = 1.0) -> None:
    """Desenha as linhas virtuais de contagem com seus totais.

    As linhas estao em pixels da imagem original; ``scale`` (processada/original)
    as leva para a imagem exibida.
    """
    for line in lines:
        p1 = tuple(int(round(v * scale)) for v in line.p1)
        p2 = tuple(int(round(v * scale)) for v in line.p2)
        cv2.line(image, p1, p2, COLOR_LINE, 2, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        _label(image, f"{line.name}: {line.count_in}/{line.count_out}", mid, COLOR_LINE, 0.5)


def draw_overlay(
    image: np.ndarray,
    result: FrameResult,
    cfg: UIConfig | None = None,
    metrics: dict | None = None,
    lines: list | None = None,
    line_scale: float = 1.0,
) -> np.ndarray:
    """Aplica todas as camadas de anotacao ao quadro (modifica ``image``)."""
    cfg = cfg or UIConfig()

    if cfg.show_vehicle_boxes:
        for vehicle in result.vehicles:
            x1, y1, x2, y2 = (int(v) for v in vehicle.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_VEHICLE, 1)

    if cfg.show_plate_boxes:
        for plate in result.plates:
            x1, y1, x2, y2 = (int(v) for v in plate.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_PLATE, 2)

    if cfg.show_tracks:
        for track in result.tracks:
            draw_track(image, track, cfg)

    if lines:
        draw_lines(image, lines, line_scale)

    if cfg.show_hud:
        draw_hud(image, result, metrics)

    return image
