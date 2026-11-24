#!/usr/bin/env python3
"""Thin CLI entrypoint to download and freeze the ETF total-return dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine.data_loader import freeze_etf_dataset, load_frozen_prices
from engine.policy import load_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze total-return ETF dataset from yfinance.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory.")
    parser.add_argument("--start", type=str, default="2004-11-20", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional exclusive end date (YYYY-MM-DD).")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data output directory.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    policy = load_policy(root / "config" / "risk_policy.json")
    data_dir = args.data_dir.resolve() if args.data_dir else (root / "data")

    csv_path = freeze_etf_dataset(
        policy.universe,
        start=args.start,
        end=args.end,
        data_dir=data_dir,
    )
    prices, manifest = load_frozen_prices(data_dir, tickers=policy.universe, verify_hash=True)
    last_date = prices.index[-1].strftime("%Y-%m-%d")
    last_spy = float(prices["SPY"].iloc[-1]) if "SPY" in prices.columns else float("nan")
    print(
        f"Froze {csv_path} | rows={manifest['rows']} | "
        f"{manifest['start_date']}..{manifest['end_date']} | "
        f"frozen_at_utc={manifest['frozen_at_utc']} | sha256={manifest['sha256']}"
    )
    print(f"Last row ({last_date}): SPY={last_spy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
