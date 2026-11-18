"""Agregacao de metricas em tempo de execucao (alimenta o HUD e o relatorio)."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

from ..core.clock import FPSCounter, Percentiles, RollingTimer


@dataclass(slots=True)
class RuntimeMetrics:
    """Fotografia instantanea do desempenho e do rendimento do pipeline."""

    fps: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    latency: Percentiles = field(default_factory=Percentiles)      # tempo de servico
    latency_e2e: Percentiles = field(default_factory=Percentiles)  # ponta a ponta (com fila)
    detect: Percentiles = field(default_factory=Percentiles)
    ocr: Percentiles = field(default_factory=Percentiles)
    track: Percentiles = field(default_factory=Percentiles)
    n_active_tracks: int = 0
    n_unique_plates: int = 0
    n_events: int = 0
    n_reads: int = 0                 # leituras com texto
    n_ocr_attempts: int = 0          # tentativas de OCR, com ou sem texto
    n_valid_syntax: int = 0
    counts_by_class: dict[str, int] = field(default_factory=dict)
    counts_by_category: dict[str, int] = field(default_factory=dict)
    n_vulnerable: int = 0

    @property
    def drop_rate(self) -> float:
        total = self.frames_processed + self.frames_dropped
        return self.frames_dropped / total if total > 0 else 0.0

    @property
    def syntax_valid_rate(self) -> float:
        return self.n_valid_syntax / self.n_reads if self.n_reads > 0 else 0.0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["drop_rate"] = self.drop_rate
        data["syntax_valid_rate"] = self.syntax_valid_rate
        return data


class MetricsCollector:
    """Acumula metricas com seguranca entre threads (pipeline escreve, UI le)."""

    def __init__(self, window: int = 300) -> None:
        self._fps = FPSCounter(window=60)
        self._latency = RollingTimer(window)
        self._e2e = RollingTimer(window)
        self._detect = RollingTimer(window)
        self._ocr = RollingTimer(window)
        self._track = RollingTimer(window)
        self._lock = threading.Lock()
        self._frames = 0
        self._dropped = 0
        self._events = 0
        self._reads = 0
        self._attempts = 0
        self._valid = 0
        self._plates: dict[str, int] = {}
        self._active_tracks = 0
        self._by_class: dict[str, int] = {}
        self._by_category: dict[str, int] = {}
        self._counted_tracks: set[int] = set()
        self._vulnerable = 0

    def on_track_seen(self, track_id: int, object_class: str, category: str,
                      vulnerable: bool) -> None:
        """Registra uma trilha nova. Conta objetos UNICOS, nao quadro-a-quadro.

        Somar por quadro contaria o mesmo carro 200 vezes; a estatistica que
        interessa e quantos objetos distintos passaram pela cena.
        """
        with self._lock:
            if track_id in self._counted_tracks:
                return
            self._counted_tracks.add(track_id)
            self._by_class[object_class] = self._by_class.get(object_class, 0) + 1
            self._by_category[category] = self._by_category.get(category, 0) + 1
            if vulnerable:
                self._vulnerable += 1

    def on_frame(
        self,
        total_ms: float,
        detect_ms: float = 0.0,
        ocr_ms: float = 0.0,
        track_ms: float = 0.0,
        dropped: int = 0,
        n_active_tracks: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        self._fps.tick()
        self._latency.add(total_ms)
        if latency_ms > 0:
            self._e2e.add(latency_ms)
        if detect_ms > 0:
            self._detect.add(detect_ms)
        if ocr_ms > 0:
            self._ocr.add(ocr_ms)
        if track_ms > 0:
            self._track.add(track_ms)
        with self._lock:
            self._frames += 1
            self._dropped = max(self._dropped, dropped)
            self._active_tracks = n_active_tracks

    def current_fps(self) -> float:
        """So a vazao corrente, sem calcular percentis (barato o bastante por quadro)."""
        return self._fps.fps

    def on_read(self, valid_syntax: bool) -> None:
        """Leitura que produziu texto."""
        with self._lock:
            self._reads += 1
            self._valid += int(valid_syntax)

    def on_attempt(self) -> None:
        """Tentativa de OCR concluida (com ou sem texto)."""
        with self._lock:
            self._attempts += 1

    def on_event(self, plate_text: str, replaces: str | None = None) -> None:
        """Uma decisao publicada; ``replaces`` = texto revisado (nao conta novo evento)."""
        with self._lock:
            if replaces is None:
                self._events += 1
            elif replaces in self._plates:
                self._plates[replaces] -= 1
                if self._plates[replaces] == 0:
                    del self._plates[replaces]
            if plate_text:
                self._plates[plate_text] = self._plates.get(plate_text, 0) + 1

    def snapshot(self) -> RuntimeMetrics:
        with self._lock:
            frames, dropped = self._frames, self._dropped
            events, reads, valid = self._events, self._reads, self._valid
            attempts = self._attempts
            unique, active = len(self._plates), self._active_tracks
            by_class = dict(self._by_class)
            by_category = dict(self._by_category)
            vulnerable = self._vulnerable
        return RuntimeMetrics(
            fps=self._fps.fps,
            frames_processed=frames,
            frames_dropped=dropped,
            latency=self._latency.stats(),
            latency_e2e=self._e2e.stats(),
            detect=self._detect.stats(),
            ocr=self._ocr.stats(),
            track=self._track.stats(),
            n_active_tracks=active,
            n_unique_plates=unique,
            n_events=events,
            n_reads=reads,
            n_ocr_attempts=attempts,
            n_valid_syntax=valid,
            counts_by_class=by_class,
            counts_by_category=by_category,
            n_vulnerable=vulnerable,
        )

    def reset(self) -> None:
        for timer in (self._latency, self._e2e, self._detect, self._ocr, self._track):
            timer.reset()
        with self._lock:
            self._frames = self._dropped = self._events = self._reads = self._valid = 0
            self._attempts = 0
            self._vulnerable = 0
            self._plates.clear()
            self._by_class.clear()
            self._by_category.clear()
            self._counted_tracks.clear()
