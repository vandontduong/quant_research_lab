"""Cycle 3 Candidate: Top-K cross-sectional momentum with inverse-volatility weighting.

Strictly causal: on each scheduled rebalance bar `t`, ranks all universe assets
(including defensive SHY) by trailing `mom_window`-day total return, selects the
top `top_k` assets, and weights them inversely by trailing `vol_window`-day realized
volatility (subject to `min_vol_floor` and `max_weight` <= 0.45 so drifted weights
never breach the 0.50 policy ceiling). Remains 100% invested — no 0%-yield cash drag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v003_inv_vol_top_momentum"
DEFAULT_PARAMS: dict[str, Any] = {
    "mom_window": 126,
    "vol_window": 63,
    "top_k": 3,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
HYPOTHESIS = (
    "Avoid Cycle 2's binary SMA whipsaw and 0%-cash drag by ranking all 6 universe ETFs "
    "(including SHY) by trailing 126-day momentum and inverse-volatility weighting the top 3 "
    "assets (capped at 45% each), naturally rotating into SHY/GLD during bear regimes while "
    "staying fully invested in trend leaders during bull regimes."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal top-K momentum + inverse-volatility target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    mom_window = int(cfg["mom_window"])
    vol_window = int(cfg["vol_window"])
    top_k = int(cfg["top_k"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    if top_k < 1 or top_k > n_assets:
        raise ValueError(f"top_k must be between 1 and {n_assets}")

    rets = prices.pct_change().fillna(0.0)
    rolling_vol = rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    rolling_mom = prices.pct_change(mom_window)

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
                # Select top_k indices by trailing momentum (deterministic tie-break via mergesort).
                order = np.argsort(-moms, kind="mergesort")
                chosen = order[:top_k]
                raw = np.zeros(n_assets, dtype=float)
                floored = np.maximum(vols[chosen], min_vol_floor)
                raw[chosen] = 1.0 / floored
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
