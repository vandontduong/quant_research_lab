"""Cycle 5 Candidate: Inverse-volatility weighting with smooth excess-trend tilt (100% invested).

Strictly causal: on each scheduled rebalance bar `t`, computes trailing `vol_window`-day
inverse-volatility weights across all 6 ETFs, then tilts each asset smoothly by
`exp(tilt_strength * clip(excess_z_i, -1.5, 1.5))`, where `excess_z_i` is the asset's
trailing `mom_window`-day total return minus SHY's trailing return, scaled by its
trailing `mom_window`-horizon volatility. Stays 100% invested (no 0%-cash drag) and
retains full 6-asset diversification without binary SMA whipsaws.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v005_soft_trend_inv_vol"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "mom_window": 126,
    "tilt_strength": 0.50,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
    "cash_proxy": "SHY",
}
HYPOTHESIS = (
    "Replace Cycle 2's binary SMA cash-cutoff with a smooth volatility-scaled excess-return "
    "tilt over SHY (`exp(0.5 * clip(z_excess, -1.5, 1.5))`) on top of v001 inverse-volatility "
    "weights while staying 100% invested, reducing 2022 rate-shock drawdown without cash drag or whipsaws."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal smooth-trend-tilted inverse-volatility target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    mom_window = int(cfg["mom_window"])
    tilt = float(cfg["tilt_strength"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])
    cash_proxy = str(cfg["cash_proxy"]).upper()

    cols = list(prices.columns)
    cash_idx = cols.index(cash_proxy) if cash_proxy in cols else -1

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    rolling_vol = rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    rolling_mom = prices.pct_change(mom_window)
    horizon_scale = np.sqrt(mom_window / 252.0)

    ret_arr = rets.to_numpy(dtype=float)
    vol_arr = rolling_vol.to_numpy(dtype=float)
    mom_arr = rolling_mom.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            vols = vol_arr[i]
            moms = mom_arr[i]
            if np.all(np.isfinite(vols)) and np.all(np.isfinite(moms)):
                floored = np.maximum(vols, min_vol_floor)
                cash_ret = moms[cash_idx] if cash_idx >= 0 else 0.0
                excess_ret = moms - cash_ret
                z_score = np.clip(excess_ret / (floored * horizon_scale), -1.5, 1.5)
                if cash_idx >= 0:
                    z_score[cash_idx] = 0.0
                raw = (1.0 / floored) * np.exp(tilt * z_score)
                current = _cap_and_normalize(raw, max_weight)
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
