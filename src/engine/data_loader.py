"""Versioned, frozen market dataset loader (total-return auto-adjusted ETF daily bars)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def freeze_etf_dataset(
    tickers: list[str],
    *,
    start: str = "2004-11-20",
    end: str | None = None,
    data_dir: str | Path = "data",
) -> Path:
    """Download auto-adjusted (total return) daily close prices and freeze to CSV + manifest.json."""
    import yfinance as yf

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "etf_daily_total_return.csv"
    manifest_path = out_dir / "manifest.json"

    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist is None or hist.empty:
            raise RuntimeError(f"yfinance returned no rows for {ticker}")
        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        series_map[ticker.upper()] = close.rename(ticker.upper())

    df = pd.concat(series_map.values(), axis=1).sort_index().dropna(how="any")
    df.index.name = "date"
    df.to_csv(csv_path, date_format="%Y-%m-%d")

    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "dataset": csv_path.name,
        "sha256": sha256,
        "source": "yfinance",
        "return_basis": "total_return (auto_adjust=True, split and dividend adjusted)",
        "corporate_action_notes": "Prices are split- and dividend-adjusted closes; common intersection across all tickers.",
        "tickers": list(df.columns),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "rows": len(df),
        "frozen_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path


def load_frozen_prices(
    data_dir: str | Path = "data",
    *,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    verify_hash: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load frozen daily total-return price matrix and verify manifest SHA-256."""
    d = Path(data_dir)
    csv_path = d / "etf_daily_total_return.csv"
    manifest_path = d / "manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Frozen dataset not found in {d}. Run freeze_etf_dataset() first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify_hash:
        actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if actual_hash != manifest.get("sha256"):
            raise RuntimeError(
                f"Dataset integrity check failed: {csv_path.name} hash mismatch "
                f"(expected {manifest.get('sha256')}, got {actual_hash})."
            )

    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    if tickers is not None:
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Tickers missing from frozen dataset: {missing}")
        df = df[tickers]
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if df.isna().any().any():
        raise ValueError("Frozen price matrix contains NaN values.")
    return df, manifest
