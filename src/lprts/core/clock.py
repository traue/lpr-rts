"""Medicao de tempo e vazao do pipeline (FPS, latencia p50/p95/p99).

O sistema e de tempo real, entao *medir* e parte do produto: estes contadores
alimentam tanto o HUD da interface quanto as tabelas exportadas para analise.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Percentiles:
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    max: float = 0.0
    n: int = 0


def percentiles(values: list[float]) -> Percentiles:
    """Percentis por interpolacao linear (mesma convencao de ``numpy.percentile``)."""
    if not values:
        return Percentiles()
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return Percentiles(
        p50=q(0.50), p90=q(0.90), p95=q(0.95), p99=q(0.99),
        mean=sum(ordered) / n, max=ordered[-1], n=n,
    )


class RollingTimer:
    """Janela deslizante de amostras de latencia, segura para uso entre threads."""

    def __init__(self, window: int = 300) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._samples.append(float(ms))

    def stats(self) -> Percentiles:
        with self._lock:
            values = list(self._samples)
        return percentiles(values)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class FPSCounter:
    """FPS instantaneo por janela deslizante de carimbos de tempo."""

    def __init__(self, window: int = 60) -> None:
        self._ts: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total = 0

    def tick(self, when: float | None = None) -> None:
        with self._lock:
            self._ts.append(when if when is not None else time.monotonic())
            self._total += 1

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._ts) < 2:
                return 0.0
            span = self._ts[-1] - self._ts[0]
            return (len(self._ts) - 1) / span if span > 1e-9 else 0.0

    @property
    def total(self) -> int:
        with self._lock:
            return self._total


class Stopwatch:
    """Cronometro de contexto: ``with Stopwatch() as sw: ...`` -> ``sw.ms``."""

    __slots__ = ("_t0", "ms")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
