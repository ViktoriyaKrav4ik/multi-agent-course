"""Додаємо корінь проєкту та tests/ до import path для pytest."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_tests_dir = Path(__file__).resolve().parent
for p in (_root, _tests_dir):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
