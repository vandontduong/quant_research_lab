"""Cycle 1 Candidate: Trailing inverse-volatility weighting (risk-parity approximation).

Strictly causal: on each scheduled rebalance bar, computes trailing `vol_window`-day
realized volatility using returns strictly up through bar `t`, allocates weights
proportional to 1 / vol_i (with a minimum vol floor to prevent SHY from absorbing
excessive weight, and iterative capping at `max_weight` <= 0.50).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATEGY_NAME = "v001_inv_vol_monthly"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
HYPOTHESIS = (
    "Replace static equal-weight with trailing 63-day inverse-volatility weighting "
    "(with 4% annualized vol floor and 45% target single-asset cap, trimmed if drift > 49.5%) "
    "to equalize risk contributions and reduce drawdown clustering during equity/rate volatility spikes."
)


def _cap_and_normalize(raw_weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Iteratively cap weights at `max_weight` while keeping sum == 1.0."""
    w = np.maximum(raw_weights.astype(float), 0.0)
    total = float(w.sum())
    if total <= 0:
        return np.full_like(w, 1.0 / len(w))
    w = w / total
    for _ in range(len(w) + 2):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] += excess * (w[under] / under_sum)
    return w


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal inverse-volatility target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    rolling_vol = rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)

    ret_arr = rets.to_numpy(dtype=float)
    vol_arr = rolling_vol.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            vols = vol_arr[i]
            if np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                inv_v = 1.0 / floored
                current = _cap_and_normalize(inv_v, max_weight)
            else:
                current = np.full(n_assets, 1.0 / n_assets, dtype=float)
        else:
            grown = current * (1.0 + ret_arr[i])
            total = float(grown.sum())
            current = grown / total if total > 0 else np.full(n_assets, 1.0 / n_assets, dtype=float)
            if float(current.max()) > 0.495:
                current = _cap_and_normalize(current, 0.49)
        w_arr[i] = current

    return pd.DataFrame(w_arr, index=prices.index, columns=prices.columns)
