"""Batch 2 / Cycle 15 (Paper #10 — Deep RL Portfolio Management / EIIE, arXiv:1706.10059).

Adapts Jiang, Xu & Liang (2017) to long-only daily ETFs:
- Ensemble of Identical Independent Evaluators (EIIE): applies the exact same shared
  evaluator parameter vector `phi` independently to each asset's trailing price-relative
  and volatility feature tensor `[r_5d, r_21d, r_63d, inv_vol_z]`, plus a previous-weight
  transaction-cost remainder channel (`prev_w`) that directly penalizes unnecessary churn.
- Updates `phi` strictly causally via online policy ascent on realized net log-wealth
  (after 5 bps/side transaction cost remainder factor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v015_eiie_portfolio_rl"
STRATEGY_NOTE = "EIIE Portfolio RL (arXiv:1706.10059): shared per-asset evaluator + prev-weight cost-remainder softmax."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "prev_w_bias": 1.20,
    "learning_rate": 0.08,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "prev_w_bias": 0.60},
    {**DEFAULT_PARAMS, "prev_w_bias": 1.80},
    {**DEFAULT_PARAMS, "learning_rate": 0.04},
    {**DEFAULT_PARAMS, "learning_rate": 0.14},
]
HYPOTHESIS = (
    "Deep RL Portfolio Management / EIIE (arXiv:1706.10059): applying identical shared evaluator "
    "weights across all 6 ETFs with an explicit previous-weight transaction-cost remainder channel "
    "and online net log-wealth updates should adaptively reallocate while controlling churn."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal EIIE shared-evaluator portfolio weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    prev_w_bias = float(cfg["prev_w_bias"])
    lr = float(cfg["learning_rate"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    r5 = np.nan_to_num(prices.pct_change(5).to_numpy(dtype=float), nan=0.0)
    r21 = np.nan_to_num(prices.pct_change(21).to_numpy(dtype=float), nan=0.0)
    r63 = np.nan_to_num(prices.pct_change(63).to_numpy(dtype=float), nan=0.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)

    # Shared EIIE evaluator parameters over 4 per-asset channels: [r5_z, r21_z, r63_z, inv_vol_z]
    phi = np.array([0.0, 0.05, 0.05, 0.80], dtype=float)
    last_rebal_idx: int | None = None
    last_x: np.ndarray | None = None
    last_w: np.ndarray | None = None
    min_bars = max(vol_window, 63)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)

                # Causal EIIE online update on net log-wealth over realized period (last_rebal_idx, i]
                if last_rebal_idx is not None and last_x is not None and last_w is not None:
                    period_r = ret_arr[last_rebal_idx + 1 : i + 1]
                    if len(period_r) > 0:
                        asset_rel = np.prod(1.0 + period_r, axis=0)
                        port_rel = float(np.dot(last_w, asset_rel))
                        if port_rel > 1e-8:
                            # Exact gradient of log(w · asset_rel) w.r.t. softmax logits:
                            # dL/dz_j = w_j * (asset_rel_j / port_rel - 1)
                            d_logits = last_w * (asset_rel / port_rel - 1.0)
                            grad_phi = d_logits @ last_x  # (4,)
                            phi = np.clip(0.97 * phi + lr * grad_phi, -1.5, 2.5)
                            # Keep inverse-vol prior positive for stability
                            phi[3] = max(phi[3], 0.30)

                inv_v = 1.0 / floored
                inv_v_z = (inv_v - float(inv_v.mean())) / (float(inv_v.std()) + 1e-8)
                x_i = np.stack(
                    [
                        np.tanh(r5[i] / (floored * np.sqrt(5.0 / 252.0))),
                        np.tanh(r21[i] / (floored * np.sqrt(21.0 / 252.0))),
                        np.tanh(r63[i] / (floored * np.sqrt(63.0 / 252.0))),
                        np.clip(inv_v_z, -2.0, 2.0),
                    ],
                    axis=-1,
                )  # (n_assets, 4)

                # Shared EIIE logits + previous-weight transaction-cost remainder channel
                logits = (x_i @ phi) + prev_w_bias * current
                logits = logits - float(logits.max())
                raw_w = np.exp(logits)
                new_w = _cap_and_normalize(raw_w, max_weight)

                last_rebal_idx = i
                last_x = x_i
                last_w = new_w.copy()
                current = new_w
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
