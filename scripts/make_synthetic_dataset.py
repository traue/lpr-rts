#!/usr/bin/env python3
"""Ponto de entrada compativel; o gerador e distribuido no pacote (``lprts.synthetic``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lprts.synthetic import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
