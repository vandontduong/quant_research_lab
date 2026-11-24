#!/usr/bin/env python3
"""Explicit opt-in CLI for one-time locked holdout audit (never invoked by run_experiment.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine.runner import run_holdout_audit
from tools.experiment_cli import load_strategy_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-time sealed holdout audit (requires --confirm-holdout-audit)."
    )
    parser.add_argument("--strategy", type=Path, required=True, help="Path to strategy module.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--confirm-holdout-audit",
        action="store_true",
        help="Explicitly confirm unlocking the sealed holdout window for a one-time audit.",
    )
    args = parser.parse_args(argv)

    if not args.confirm_holdout_audit:
        print(
            "ERROR: Holdout window is locked. Pass --confirm-holdout-audit to run a one-time holdout audit.",
            file=sys.stderr,
        )
        return 2

    root = args.root.resolve()
    strategy_path = args.strategy if args.strategy.is_absolute() else (root / args.strategy)
    mod = load_strategy_module(strategy_path)
    strategy_name = str(getattr(mod, "STRATEGY_NAME", strategy_path.stem))
    params = dict(getattr(mod, "DEFAULT_PARAMS", {}))

    audit = run_holdout_audit(
        strategy_fn=mod.generate_weights,
        params=params,
        strategy_name=strategy_name,
        root_dir=root,
        output_path=args.output,
        confirm_holdout_audit=True,
    )
    cand = audit["candidate"]["holdout_1x"]
    print(
        f"Holdout audit ({strategy_name}): {cand['start_date']}..{cand['end_date']} | "
        f"Sharpe={cand['sharpe']:.3f} (raw={cand['raw_sharpe']:.3f}) | CAGR={cand['cagr']*100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
