"""Batch 2 / Cycle 14 (Paper #5 — Deep Learning StatArb, arXiv:2106.04028).

Adapts Guijarro-Ordonez, Pelger & Zanotti (2021) to long-only daily ETFs:
- Stage 1 (PCA Residual Extraction): extracts the leading latent market factor causally via SVD
  on trailing `pca_window`-day standardized returns and forms orthogonal residual return paths.
- Stage 2 (Multi-Scale Temporal Pattern Filter): applies convolutional-like multi-horizon
  cumulative residual z-scores (5d, 15d, 30d) to detect short-term dislocations.
- Stage 3 (Cost-Aware Policy): maps contrarian residual scores (`-z_resid`) into smooth tilts
  on top of inverse-volatility weights, applying a turnover deadband (`trade_band`) so tiny
  signals do not incur 5 bps/side transaction costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v014_stat_arb_residual_mr"
STRATEGY_NOTE = "Deep StatArb (arXiv:2106.04028): PCA-orthogonal residual multi-scale mean reversion with cost deadband."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "pca_window": 126,
    "mr_tilt": 0.25,
    "trade_band": 0.03,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "pca_window": 90},
    {**DEFAULT_PARAMS, "pca_window": 189},
    {**DEFAULT_PARAMS, "mr_tilt": 0.15},
    {**DEFAULT_PARAMS, "mr_tilt": 0.40},
]
HYPOTHESIS = (
    "Deep Learning StatArb (arXiv:2106.04028): extracting PCA-orthogonal residual paths and "
    "applying a multi-scale (5d/15d/30d) contrarian filter with a cost-aware rebalance deadband "
    "should capture cross-asset mean-reversion alpha without excessive turnover."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal PCA-residual mean-reversion target weights with cost deadband."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    pca_window = int(cfg["pca_window"])
    mr_tilt = float(cfg["mr_tilt"])
    trade_band = float(cfg["trade_band"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, pca_window)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                win = ret_arr[i - pca_window + 1 : i + 1]
                # Stage 1: Causal 1-factor PCA orthogonalization
                mu = win.mean(axis=0, keepdims=True)
                sd = np.maximum(win.std(axis=0, ddof=1, keepdims=True), 1e-6)
                z_win = (win - mu) / sd
                _, _, vt = np.linalg.svd(z_win, full_matrices=False)
                v1 = vt[0:1, :]  # (1, n_assets)
                proj = (z_win @ v1.T) @ v1
                resid = z_win - proj  # (pca_window, n_assets)

                # Stage 2: Multi-scale temporal residual filter (5d, 15d, 30d cumulative z-scores)
                c5 = resid[-5:].sum(axis=0) / np.sqrt(5.0)
                c15 = resid[-15:].sum(axis=0) / np.sqrt(15.0)
                c30 = resid[-30:].sum(axis=0) / np.sqrt(30.0)
                mr_score = np.clip(-(0.45 * c5 + 0.35 * c15 + 0.20 * c30), -2.0, 2.0)

                # Stage 3: Cost-aware policy with turnover deadband
                raw = (1.0 / floored) * np.exp(mr_tilt * mr_score)
                proposed = _cap_and_normalize(raw, max_weight)
                one_way_delta = float(np.abs(proposed - current).sum()) * 0.5
                if one_way_delta >= trade_band:
                    current = proposed
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
