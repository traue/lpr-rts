#!/usr/bin/env python3
"""Ponto de entrada compativel; o benchmark e distribuido no pacote (``lprts.benchmark``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lprts.benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
