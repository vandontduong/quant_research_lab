"""Cycle 0 Baseline: Monthly-rebalanced equal-weight multi-asset ETF portfolio.

Strictly causal: on each month-end trading bar (approximated by rebalance every
`rebalance_every_days` trading bars), target weights reset to equal weight across
all universe assets (1/N <= 0.50).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATEGY_NAME = "v000_equal_weight_monthly"
DEFAULT_PARAMS: dict[str, Any] = {
    "rebalance_every_days": 21,
}
HYPOTHESIS = (
    "Establish Cycle 0 naive baseline: equal-weight allocation across the 6-ETF universe "
    "(SPY, QQQ, IWM, TLT, GLD, SHY) rebalanced every 21 trading days."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal target weights for equal-weight periodic rebalancing."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    step = int(cfg["rebalance_every_days"])
    if step < 1:
        raise ValueError("rebalance_every_days must be >= 1")

    n_days, n_assets = prices.shape
    eq_w = 1.0 / n_assets
    w_arr = np.zeros((n_days, n_assets), dtype=float)
    ret_arr = prices.pct_change().fillna(0.0).to_numpy(dtype=float)

    current = np.full(n_assets, eq_w, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            current = np.full(n_assets, eq_w, dtype=float)
        else:
            # Let target weights drift naturally between scheduled rebalances so we don't trade daily.
            grown = current * (1.0 + ret_arr[i])
            total = float(grown.sum())
            current = grown / total if total > 0 else np.full(n_assets, eq_w, dtype=float)
        w_arr[i] = current

    return pd.DataFrame(w_arr, index=prices.index, columns=prices.columns)
