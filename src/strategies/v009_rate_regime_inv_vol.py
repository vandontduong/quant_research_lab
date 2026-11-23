"""Cycle 9 Candidate: Stock-bond correlation regime-adaptive inverse-volatility weighting.

Strictly causal: builds on v001 trailing 63-day inverse-volatility weights, and adds a single
macro-regime diversification adjustment: when trailing `corr_window`-day correlation between
SPY and TLT turns positive (`corr(SPY, TLT) > 0`, as in inflation/rate-shock regimes like 2022),
TLT no longer hedges equity drawdowns. Smoothly scales down TLT's raw inverse-vol score by
`1 - corr_penalty * max(0, corr(SPY, TLT))` before normalizing and capping at `max_weight`,
naturally shifting weight toward SHY and GLD without binary cutoffs or high turnover.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v009_rate_regime_inv_vol"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "corr_window": 63,
    "corr_penalty": 0.75,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "corr_window": 42},
    {**DEFAULT_PARAMS, "corr_window": 90},
    {**DEFAULT_PARAMS, "corr_penalty": 0.50},
    {**DEFAULT_PARAMS, "corr_penalty": 0.90},
]
HYPOTHESIS = (
    "v001 and Cycles 4–8 all suffered their worst stress drawdown in 2022_rate_shock (-15.4% DD) "
    "because TLT lost its equity-hedging property when stock-bond correlation turned positive. "
    "Smoothly scaling down TLT's raw inverse-vol weight by 1 - 0.75 * max(0, corr_63d(SPY, TLT)) "
    "should cut rate-shock drawdown and lift OOS Sharpe while preserving low monthly turnover."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal stock-bond correlation-adaptive inverse-volatility weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    corr_window = int(cfg["corr_window"])
    corr_penalty = float(cfg["corr_penalty"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    cols = list(prices.columns)
    spy_idx = cols.index("SPY") if "SPY" in cols else -1
    tlt_idx = cols.index("TLT") if "TLT" in cols else -1

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, corr_window)

    for i in range(n_days):
        if i % step == 0:
            if i >= min_bars:
                vol_win = ret_arr[i - vol_window + 1 : i + 1]
                vols = np.std(vol_win, axis=0, ddof=1) * np.sqrt(252.0)
                if np.all(np.isfinite(vols)):
                    floored = np.maximum(vols, min_vol_floor)
                    raw = 1.0 / floored
                    if spy_idx >= 0 and tlt_idx >= 0:
                        c_win = ret_arr[i - corr_window + 1 : i + 1]
                        s_r = c_win[:, spy_idx]
                        t_r = c_win[:, tlt_idx]
                        s_std = float(np.std(s_r, ddof=1))
                        t_std = float(np.std(t_r, ddof=1))
                        if s_std > 1e-10 and t_std > 1e-10:
                            rho = float(np.corrcoef(s_r, t_r)[0, 1])
                            if np.isfinite(rho) and rho > 0.0:
                                raw[tlt_idx] *= max(0.05, 1.0 - corr_penalty * rho)
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
