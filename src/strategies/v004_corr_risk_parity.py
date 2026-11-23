"""Cycle 4 Candidate: Correlation-aware risk parity (inverse volatility * inverse average correlation).

Strictly causal: on each scheduled rebalance bar `t`, computes trailing `cov_window`-day
realized volatility and shrunk correlation matrix across the universe. Assets that are
heavily correlated with other assets in the universe (e.g. SPY/QQQ/IWM equity cluster)
are down-weighted by `1 / (vol_i * avg_pos_corr_i)`, equalizing cluster-level risk
without unstable unconstrained matrix inversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v004_corr_aware_risk_parity"
DEFAULT_PARAMS: dict[str, Any] = {
    "cov_window": 63,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "corr_shrinkage": 0.25,
    "max_weight": 0.45,
}
HYPOTHESIS = (
    "v001 inverse-volatility ignores the ~0.90 pairwise correlation among SPY, QQQ, and IWM, "
    "over-concentrating equity risk. Weighting each asset by 1 / (vol_i * (1 + avg_pos_corr_i)) "
    "using shrunk 63-day correlations should penalize redundant correlated clusters and improve OOS Sharpe."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal correlation-aware risk-parity target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    cov_window = int(cfg["cov_window"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    shrinkage = float(cfg["corr_shrinkage"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            if i >= cov_window:
                window_r = ret_arr[i - cov_window + 1 : i + 1]
                vols = np.std(window_r, axis=0, ddof=1) * np.sqrt(252.0)
                if np.all(np.isfinite(vols)):
                    floored = np.maximum(vols, min_vol_floor)
                    corr = np.corrcoef(window_r, rowvar=False)
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                    # Shrink sample correlation matrix toward identity to reduce estimation noise.
                    shrunk_corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(n_assets)
                    # Average positive off-diagonal correlation per asset (cluster redundancy penalty).
                    np.fill_diagonal(shrunk_corr, 0.0)
                    avg_pos_corr = np.maximum(shrunk_corr, 0.0).sum(axis=1) / max(n_assets - 1, 1)
                    raw = 1.0 / (floored * (1.0 + avg_pos_corr))
                    current = _cap_and_normalize(raw, max_weight)
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
