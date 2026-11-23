#!/usr/bin/env python3
"""Thin CLI entrypoint for analyzing experiments, syncing results.tsv, and plotting progress.svg."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.analyze_experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
