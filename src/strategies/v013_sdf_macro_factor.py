"""Batch 2 / Cycle 13 (inspired by Deep Learning in Asset Pricing / SDF, arXiv:1904.00745).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (not an adversarial GAN/LSTM
SDF reproduction of Chen, Pelger & Zhu, 2019):
- ETF-derived macro proxies: constructs 4 causal state series from the 6 ETFs themselves
  (`SPY-TLT` 63d return spread, `TLT` 21d vol, `GLD-SHY` 63d spread, and cross-asset dispersion)
  rather than external macroeconomic panels.
- Closed-form ridge SDF proxy: estimates conditional expected returns `mu_cond` via trailing OLS,
  solves `w_sdf = (Sigma + lambda * I)^-1 @ mu_cond`, and blends the scores into
  inverse-volatility weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v013_sdf_macro_factor"
STRATEGY_NOTE = "Deep Asset Pricing / SDF-inspired (arXiv:1904.00745): closed-form ridge SDF on ETF macro proxies."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "train_window": 504,
    "ridge_lambda": 0.05,
    "sdf_tilt": 0.30,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "train_window": 378},
    {**DEFAULT_PARAMS, "train_window": 756},
    {**DEFAULT_PARAMS, "sdf_tilt": 0.15},
    {**DEFAULT_PARAMS, "sdf_tilt": 0.50},
]
HYPOTHESIS = (
    "Deep Learning in Asset Pricing (arXiv:1904.00745): conditioning expected asset returns on "
    "causal macro-proxy states (equity-bond spread, rate vol, gold-cash trend) and solving a "
    "regularized GMM SDF objective should tilt inverse-vol weights toward higher-pricing-kernel assets."
)


def _macro_state_matrix(prices: pd.DataFrame, rets: pd.DataFrame) -> np.ndarray:
    """Build strictly causal (n_days, 4) macro-state vector from ETF proxies."""
    cols = list(prices.columns)
    m63 = prices.pct_change(63).fillna(0.0)
    spy_tlt = (m63["SPY"] - m63["TLT"]) if ("SPY" in cols and "TLT" in cols) else pd.Series(0.0, index=prices.index)
    gld_shy = (m63["GLD"] - m63["SHY"]) if ("GLD" in cols and "SHY" in cols) else pd.Series(0.0, index=prices.index)
    tlt_vol = (
        rets["TLT"].rolling(21, min_periods=21).std(ddof=1).fillna(0.01) * np.sqrt(252.0)
        if "TLT" in cols
        else pd.Series(0.10, index=prices.index)
    )
    disp = rets.rolling(21, min_periods=21).std(ddof=1).mean(axis=1).fillna(0.01) * np.sqrt(252.0)

    raw = np.column_stack(
        [
            np.tanh(spy_tlt.to_numpy(dtype=float) * 3.0),
            np.tanh(gld_shy.to_numpy(dtype=float) * 3.0),
            np.tanh((tlt_vol.to_numpy(dtype=float) - 0.14) * 5.0),
            np.tanh((disp.to_numpy(dtype=float) - 0.12) * 5.0),
        ]
    )
    return raw


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal macro-conditioned SDF-tilted target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    train_window = int(cfg["train_window"])
    ridge = float(cfg["ridge_lambda"])
    sdf_tilt = float(cfg["sdf_tilt"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    macro = _macro_state_matrix(prices, rets)  # (n_days, 4)
    # Add intercept column -> (n_days, 5)
    macro_x = np.column_stack([np.ones(n_days, dtype=float), macro])

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, 126)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                if i >= min_bars + 63:
                    start = max(63, i - train_window)
                    # Causal ridge regression: macro_x[t] -> ret_arr[t+1] for t in [start, i-1]
                    x_tr = macro_x[start:i]
                    y_tr = ret_arr[start + 1 : i + 1]
                    k = x_tr.shape[1]
                    xtx = x_tr.T @ x_tr + (ridge * len(x_tr)) * np.eye(k)
                    beta = np.linalg.solve(xtx, x_tr.T @ y_tr)  # (5, n_assets)
                    mu_cond = macro_x[i] @ beta  # predicted daily return vector

                    # Regularized GMM SDF portfolio: w_sdf = (Sigma + lambda * I)^-1 @ mu_cond
                    cov = np.cov(y_tr, rowvar=False, ddof=1)
                    cov_reg = cov + (ridge * np.trace(cov) / n_assets + 1e-6) * np.eye(n_assets)
                    sdf_raw = np.linalg.solve(cov_reg, mu_cond)
                    sdf_std = float(np.std(sdf_raw)) + 1e-8
                    sdf_z = np.clip((sdf_raw - float(np.mean(sdf_raw))) / sdf_std, -2.0, 2.0)
                else:
                    sdf_z = np.zeros(n_assets, dtype=float)

                raw = (1.0 / floored) * np.exp(sdf_tilt * sdf_z)
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
