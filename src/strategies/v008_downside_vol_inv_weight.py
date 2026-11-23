"""Cycle 8 Candidate: Asymmetric downside-semivolatility inverse weighting.

Strictly causal: on each scheduled rebalance bar `t`, computes both trailing `vol_window`-day
total volatility and downside semideviation (annualized RMS of negative daily returns),
blends them (`downside_weight * semi_vol + (1 - downside_weight) * total_vol`), applies a
minimum vol floor, and allocates inverse-risk weights capped at `max_weight`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v008_downside_vol_inv_weight"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "downside_weight": 0.60,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "vol_window": 42},
    {**DEFAULT_PARAMS, "vol_window": 84},
    {**DEFAULT_PARAMS, "downside_weight": 0.40},
    {**DEFAULT_PARAMS, "downside_weight": 0.80},
]
HYPOTHESIS = (
    "v001 penalizes upside volatility identically to downside volatility, underweighting assets "
    "during strong upside breakouts while reacting slowly to asymmetric left-tail selloffs. "
    "Blending trailing 63-day downside semideviation (60%) with total volatility (40%) should "
    "preserve upside participation and shift faster toward defensive assets in drawdowns."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal downside-semivolatility inverse-risk target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    ds_weight = float(cfg["downside_weight"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            if i >= vol_window:
                win = ret_arr[i - vol_window + 1 : i + 1]
                tot_vol = np.std(win, axis=0, ddof=1) * np.sqrt(252.0)
                neg = np.minimum(win, 0.0)
                # Annualized downside semideviation (sqrt(2) scales half-normal to match total vol scale).
                semi_vol = np.sqrt(np.mean(neg * neg, axis=0)) * np.sqrt(252.0 * 2.0)
                blended = ds_weight * semi_vol + (1.0 - ds_weight) * tot_vol
                if np.all(np.isfinite(blended)):
                    floored = np.maximum(blended, min_vol_floor)
                    current = _cap_and_normalize(1.0 / floored, max_weight)
                else:
                    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
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
