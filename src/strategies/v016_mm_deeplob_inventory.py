"""Batch 3 / Cycle 16 (inspired by Market Making RL, arXiv:1804.04216 & DeepLOB, arXiv:1808.03668).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (no limit-order-book data,
no DeepLOB CNN-LSTM, and no RL training; Spooner et al., 2018; Zhang, Zohren & Roberts, 2019):
- Multi-horizon daily return filter: averages 3d, 10d, and 30d vol-normalized return and
  volatility-compression scores rather than ingesting order-book microstructure.
- Asymmetric deviation penalty: treats deviations from the inverse-volatility anchor as
  inventory risk and applies a hand-crafted dampening penalty when trailing volatility is
  elevated or recent returns are negative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v016_mm_deeplob_inventory"
STRATEGY_NOTE = "MM-RL + DeepLOB-inspired (arXiv:1804.04216 / 1808.03668): 3/10/30d return filter + inventory dampening."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "pressure_tilt": 0.25,
    "inventory_aversion": 0.60,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "pressure_tilt": 0.15},
    {**DEFAULT_PARAMS, "pressure_tilt": 0.40},
    {**DEFAULT_PARAMS, "inventory_aversion": 0.35},
    {**DEFAULT_PARAMS, "inventory_aversion": 0.85},
]
HYPOTHESIS = (
    "Market Making RL (arXiv:1804.04216) + DeepLOB (arXiv:1808.03668): combining multi-scale "
    "(3d/10d/30d) inception pressure features with asymmetrically dampened inventory-risk control "
    "around an inverse-vol anchor should curb trend-chasing instability and reduce turnover."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal DeepLOB-feature + asymmetric-inventory-dampened weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    pressure_tilt = float(cfg["pressure_tilt"])
    inv_aversion = float(cfg["inventory_aversion"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )

    # Multi-scale DeepLOB-style inception return filters (3d, 10d, 30d)
    r3 = np.nan_to_num(prices.pct_change(3).to_numpy(dtype=float), nan=0.0)
    r10 = np.nan_to_num(prices.pct_change(10).to_numpy(dtype=float), nan=0.0)
    r30 = np.nan_to_num(prices.pct_change(30).to_numpy(dtype=float), nan=0.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, 30)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                anchor_w = _cap_and_normalize(1.0 / floored, max_weight)

                # Multi-scale inception score (mean-reversion on 3d microstructure noise, persistence on 10d/30d)
                z3 = np.tanh(r3[i] / (floored * np.sqrt(3.0 / 252.0)))
                z10 = np.tanh(r10[i] / (floored * np.sqrt(10.0 / 252.0)))
                z30 = np.tanh(r30[i] / (floored * np.sqrt(30.0 / 252.0)))
                inception_score = -0.25 * z3 + 0.35 * z10 + 0.40 * z30

                # Asymmetrically dampened speculative tilt: penalize positive speculation more than defensive cuts
                dampened_score = np.where(
                    inception_score > 0.0,
                    inception_score * (1.0 - 0.5 * inv_aversion),
                    inception_score,
                )
                target_w = _cap_and_normalize(anchor_w * np.exp(pressure_tilt * dampened_score), max_weight)

                # Inventory-risk mean reversion: blend toward target, penalized by current drift from anchor
                inv_dev = np.abs(current - anchor_w).sum()
                blend_alpha = max(0.35, 1.0 - inv_aversion * min(inv_dev, 0.8))
                blended = blend_alpha * target_w + (1.0 - blend_alpha) * anchor_w
                current = _cap_and_normalize(blended, max_weight)
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
