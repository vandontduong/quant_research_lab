#!/usr/bin/env python3
"""Thin CLI entrypoint for running a single quant_research_lab experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.experiment_cli import run_experiment_cli

if __name__ == "__main__":
    raise SystemExit(run_experiment_cli())
