"""Configuracao de logging com saida colorida via ``rich`` (degrada para stdlib)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    """Configura o logger raiz uma unica vez por processo."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    handlers: list[logging.Handler] = []
    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(rich_tracebacks=True, show_path=False, markup=False))
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover - fallback sem rich
        handlers.append(logging.StreamHandler(sys.stderr))
        fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level.upper(), format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    # Bibliotecas de terceiros sao verbosas demais em DEBUG.
    for noisy in ("ultralytics", "PIL", "matplotlib", "urllib3", "easyocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
