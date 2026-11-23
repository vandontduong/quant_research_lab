"""Cycle 7 Candidate: Equal Risk Contribution (ERC / True Risk Parity) on shrunk covariance.

Strictly causal: on each scheduled rebalance bar `t`, estimates the trailing `cov_window`-day
covariance matrix, shrinks off-diagonal covariances toward the diagonal (`shrinkage`),
applies a minimum volatility floor, and solves for Equal Risk Contribution (ERC) weights
where each asset's marginal risk contribution `w_i * (Sigma @ w)_i` is equalized via
deterministic multiplicative fixed-point updates (capped at `max_weight`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v007_erc_risk_parity"
DEFAULT_PARAMS: dict[str, Any] = {
    "cov_window": 126,
    "shrinkage": 0.20,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "cov_window": 90},
    {**DEFAULT_PARAMS, "cov_window": 168},
    {**DEFAULT_PARAMS, "shrinkage": 0.10},
    {**DEFAULT_PARAMS, "shrinkage": 0.35},
]
HYPOTHESIS = (
    "v001 inverse-volatility ignores cross-asset covariances and v004 only used a heuristic "
    "correlation penalty. Solving for true Equal Risk Contribution (ERC) weights on a shrunk "
    "126-day covariance matrix equalizes marginal risk contributions across the correlated "
    "SPY/QQQ/IWM equity cluster vs TLT/GLD/SHY, improving OOS Sharpe without inflating turnover."
)


def _solve_erc_weights(cov: np.ndarray, max_weight: float, n_iter: int = 40) -> np.ndarray:
    """Deterministic multiplicative fixed-point solver for long-only capped ERC weights."""
    n = cov.shape[0]
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    w = _cap_and_normalize(1.0 / vols, max_weight)
    for _ in range(n_iter):
        mrc = cov @ w
        rc = w * mrc
        total_var = float(rc.sum())
        if total_var <= 1e-14 or not np.all(np.isfinite(rc)):
            break
        target_rc = total_var / n
        ratio = target_rc / np.maximum(rc, 1e-14)
        # Damped multiplicative update guarantees smooth convergence without oscillation.
        w_next = w * np.power(ratio, 0.5)
        w = _cap_and_normalize(w_next, max_weight)
    return w


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal Equal Risk Contribution (ERC) target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    cov_window = int(cfg["cov_window"])
    shrinkage = float(cfg["shrinkage"])
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
            if i >= cov_window:
                window_r = ret_arr[i - cov_window + 1 : i + 1]
                cov = np.cov(window_r, rowvar=False, ddof=1) * 252.0
                if np.all(np.isfinite(cov)):
                    vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
                    floored_vols = np.maximum(vols, min_vol_floor)
                    corr = cov / np.outer(vols, vols)
                    corr = np.clip(np.nan_to_num(corr, nan=0.0), -0.99, 0.99)
                    np.fill_diagonal(corr, 1.0)
                    shrunk_corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(n_assets)
                    shrunk_cov = shrunk_corr * np.outer(floored_vols, floored_vols)
                    current = _solve_erc_weights(shrunk_cov, max_weight)
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
