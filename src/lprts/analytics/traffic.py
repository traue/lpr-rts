"""Analitica de trafego: linhas virtuais, sentido de fluxo e velocidade.

Tres medidas derivadas da trajetoria de cada trilha:

* **Contagem por linha virtual** - o operador desenha segmentos na imagem; conta-se
  quando o ponto de contato com o solo cruza cada segmento, com o sentido do
  cruzamento.
* **Direcao do movimento** - deslocamento liquido do centro da caixa ao longo
  da janela recente, expresso como direcao **na imagem** (direita, esquerda,
  baixo, cima). Nao sao pontos cardeais: "baixo" quase sempre significa
  "aproximando-se da camera", mas isso depende da instalacao.
* **Velocidade** - com homografia calibrada, o deslocamento do ponto de contato
  com o solo vira metros e a velocidade sai em km/h. Sem calibracao o campo
  ``speed_kmh`` fica nulo - preferimos ausencia de dado a um numero sem
  significado fisico.

Convencoes que a validade das medidas depende:

* **Tempo de midia.** As trilhas carregam tempos do video (ou de chegada, ao
  vivo), nunca do relogio de processamento - senao a velocidade dependeria da
  maquina.
* **Coordenadas originais.** Linhas e calibracao sao desenhadas na imagem
  original; as trilhas vivem na imagem processada. ``coord_scale`` (processada
  / original) converte uma na outra.

Nota de validade: a homografia assume que o ponto medido esta *sobre o plano
do solo*. Usamos o meio da borda inferior da caixa (contato pneu-solo). No modo
so-placa a caixa e a da placa, que fica ~0,4-0,6 m acima do solo: ali a
velocidade tem vies sistematico e deve ser tratada como aproximacao.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import TrackState
from ..utils.geometry import (
    apply_homography,
    crossing_direction,
    homography_from_points,
    segments_intersect,
)

DIRECOES = ("direita", "esquerda", "baixo", "cima", "parado")


@dataclass(slots=True)
class VirtualLine:
    """Segmento de contagem definido em coordenadas da imagem ORIGINAL."""

    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    count_in: int = 0
    count_out: int = 0

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    def as_dict(self) -> dict:
        return {
            "name": self.name, "p1": list(self.p1), "p2": list(self.p2),
            "count_in": self.count_in, "count_out": self.count_out, "total": self.total,
        }


@dataclass(slots=True)
class CrossingEvent:
    track_id: int
    line_name: str
    direction: str
    frame_index: int
    video_time_s: float
    speed_kmh: float | None = None


@dataclass(slots=True)
class Calibration:
    """Calibracao metrica da cena (pontos em pixels da imagem ORIGINAL).

    ``image_points``/``world_points``: 4+ correspondencias entre pixels e metros
    no plano do solo (por exemplo, os cantos de um retangulo medido no asfalto,
    ou as marcas de faixa, cujo comprimento e padronizado). ``image_size`` e a
    resolucao em que os pontos foram marcados; serve para detectar uso com uma
    fonte de outra resolucao.
    """

    image_points: list[tuple[float, float]] = field(default_factory=list)
    world_points: list[tuple[float, float]] = field(default_factory=list)
    meters_per_pixel: float | None = None
    image_size: tuple[int, int] | None = None
    _h: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.build()

    def build(self) -> np.ndarray | None:
        if len(self.image_points) >= 4 and len(self.image_points) == len(self.world_points):
            self._h = homography_from_points(
                np.asarray(self.image_points, dtype=np.float64),
                np.asarray(self.world_points, dtype=np.float64),
            )
        return self._h

    @property
    def homography(self) -> np.ndarray | None:
        return self._h

    @property
    def is_metric(self) -> bool:
        """True se houver como converter pixels em metros."""
        return self._h is not None or self.meters_per_pixel is not None

    def to_world(self, pts: np.ndarray) -> np.ndarray | None:
        if self._h is not None:
            return apply_homography(self._h, pts)
        if self.meters_per_pixel is not None:
            return np.asarray(pts, dtype=np.float64) * self.meters_per_pixel
        return None

    def as_dict(self) -> dict:
        return {
            "image_points": [list(p) for p in self.image_points],
            "world_points": [list(p) for p in self.world_points],
            "meters_per_pixel": self.meters_per_pixel,
            "image_size": list(self.image_size) if self.image_size else None,
        }


def ground_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Ponto de contato com o solo: meio da borda inferior da caixa."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class TrafficAnalyzer:
    """Mantem linhas virtuais e calcula direcao/velocidade das trilhas."""

    def __init__(
        self,
        lines: list[VirtualLine] | None = None,
        calibration: Calibration | None = None,
        speed_window: int = 10,
        min_points_for_direction: int = 5,
        min_displacement_px: float = 3.0,
    ) -> None:
        self.lines = lines or []
        self.calibration = calibration or Calibration()
        self.speed_window = speed_window
        self.min_points_for_direction = min_points_for_direction
        self.min_displacement_px = min_displacement_px
        self._prev_point: dict[int, tuple[float, float]] = {}

    # Compatibilidade: o nome antigo ainda e consultado por testes e diagnostico.
    @property
    def _prev_centroid(self) -> dict[int, tuple[float, float]]:
        return self._prev_point

    # -- linhas virtuais ---------------------------------------------------

    def add_line(self, name: str, p1: tuple[float, float], p2: tuple[float, float]) -> VirtualLine:
        line = VirtualLine(name, p1, p2)
        self.lines.append(line)
        return line

    def check_crossings(
        self, track: TrackState, frame_index: int, video_time_s: float, coord_scale: float = 1.0
    ) -> list[CrossingEvent]:
        """Detecta cruzamentos desde o ultimo ponto medido; conta cada linha uma vez."""
        events: list[CrossingEvent] = []
        s = coord_scale if coord_scale > 0 else 1.0
        gx, gy = ground_point(track.bbox)
        curr = (gx / s, gy / s)
        prev = self._prev_point.get(track.track_id)
        self._prev_point[track.track_id] = curr
        if prev is None:
            return events

        for line in self.lines:
            if line.name in track.crossed_lines:
                continue  # ja contado: evita contagem dupla no vaivem do detector
            if not segments_intersect(prev, curr, line.p1, line.p2):
                continue
            direction = crossing_direction(prev, curr, line.p1, line.p2)
            if direction == "in":
                line.count_in += 1
            else:
                line.count_out += 1
            track.crossed_lines.add(line.name)
            events.append(
                CrossingEvent(
                    track_id=track.track_id, line_name=line.name, direction=direction,
                    frame_index=frame_index, video_time_s=video_time_s, speed_kmh=track.speed_kmh,
                )
            )
        return events

    # -- direcao e velocidade ---------------------------------------------

    def estimate_direction(self, track: TrackState) -> str | None:
        """Direcao dominante NA IMAGEM: deslocamento liquido na janela recente."""
        pts = track.centroids[-self.speed_window :]
        if len(pts) < self.min_points_for_direction:
            return None
        arr = np.asarray(pts, dtype=np.float64)
        delta = arr[-1] - arr[0]
        if np.linalg.norm(delta) < self.min_displacement_px:
            return "parado"
        dx, dy = float(delta[0]), float(delta[1])
        if abs(dx) >= abs(dy):
            return "direita" if dx > 0 else "esquerda"
        return "baixo" if dy > 0 else "cima"

    def estimate_speed(self, track: TrackState, coord_scale: float = 1.0) -> float | None:
        """Velocidade em km/h na janela recente; ``None`` sem calibracao metrica.

        Usa o ponto de contato com o solo, em coordenadas originais, e o tempo de
        midia. A mediana das velocidades instantaneas resiste a um salto do
        detector, que distorceria a media.
        """
        if not self.calibration.is_metric:
            return None
        pts = track.ground_points[-self.speed_window :] or track.centroids[-self.speed_window :]
        times = track.timestamps[-self.speed_window :]
        if len(pts) < 3 or len(times) != len(pts):
            return None

        s = coord_scale if coord_scale > 0 else 1.0
        world = self.calibration.to_world(np.asarray(pts, dtype=np.float64) / s)
        if world is None:
            return None

        speeds: list[float] = []
        for i in range(1, len(world)):
            dt = times[i] - times[i - 1]
            if dt <= 1e-6:
                continue
            dist_m = float(np.linalg.norm(world[i] - world[i - 1]))
            speeds.append(dist_m / dt * 3.6)
        if not speeds:
            return None
        return float(np.median(speeds))

    def update(
        self,
        tracks: list[TrackState],
        frame_index: int,
        video_time_s: float,
        coord_scale: float = 1.0,
    ) -> list[CrossingEvent]:
        """Atualiza direcao/velocidade das trilhas MEDIDAS e retorna cruzamentos."""
        events: list[CrossingEvent] = []
        for track in tracks:
            track.direction = self.estimate_direction(track) or track.direction
            speed = self.estimate_speed(track, coord_scale)
            if speed is not None:
                track.speed_kmh = speed
            events.extend(self.check_crossings(track, frame_index, video_time_s, coord_scale))
        return events

    def forget(self, track_ids: set[int]) -> None:
        """Libera o estado de trilhas encerradas (execucao 24/7 nao pode vazar)."""
        for tid in track_ids:
            self._prev_point.pop(tid, None)

    def counts(self) -> dict[str, dict]:
        return {line.name: line.as_dict() for line in self.lines}

    def reset_counts(self) -> None:
        for line in self.lines:
            line.count_in = line.count_out = 0
