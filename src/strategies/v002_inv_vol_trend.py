"""Cycle 2 Candidate: Inverse-volatility weighting + trailing SMA trend filter into SHY/cash.

Strictly causal: on each scheduled rebalance bar `t`, computes trailing `vol_window`-day
inverse-volatility weights across the universe, then filters each risky asset
(all tickers except SHY) by requiring close[t] > SMA(trend_window)[t]. Risky assets
below their trailing SMA rotate into SHY up to `max_weight`, with any excess held in
cash (weight 0) so single-asset concentration never breaches the 0.50 policy ceiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v002_inv_vol_trend_filter"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "trend_window": 126,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
    "defensive_ticker": "SHY",
}
HYPOTHESIS = (
    "Add a single 126-day SMA trend filter to v001 inverse-volatility weighting: "
    "risky assets trading below their trailing 126-day SMA rotate into SHY (capped at 45%) "
    "and cash, targeting the 2022 simultaneous stock/bond drawdown (Fold 7)."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal trend-filtered inverse-volatility target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    trend_window = int(cfg["trend_window"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])
    defensive = str(cfg["defensive_ticker"]).upper()

    cols = list(prices.columns)
    def_idx = cols.index(defensive) if defensive in cols else -1

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    rolling_vol = rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    rolling_sma = prices.rolling(trend_window, min_periods=trend_window).mean()

    px_arr = prices.to_numpy(dtype=float)
    ret_arr = rets.to_numpy(dtype=float)
    vol_arr = rolling_vol.to_numpy(dtype=float)
    sma_arr = rolling_sma.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    for i in range(n_days):
        if i % step == 0:
            vols = vol_arr[i]
            smas = sma_arr[i]
            if np.all(np.isfinite(vols)) and np.all(np.isfinite(smas)):
                floored = np.maximum(vols, min_vol_floor)
                base_w = _cap_and_normalize(1.0 / floored, max_weight)
                filtered = np.zeros(n_assets, dtype=float)
                freed = 0.0
                for j in range(n_assets):
                    if j == def_idx or px_arr[i, j] >= smas[j]:
                        filtered[j] = base_w[j]
                    else:
                        freed += base_w[j]
                if def_idx >= 0 and freed > 0:
                    room = max(0.0, max_weight - filtered[def_idx])
                    filtered[def_idx] += min(freed, room)
                    # Any remaining freed weight stays in cash (0% yield, total weight < 1.0).
                current = filtered
            else:
                current = np.full(n_assets, 1.0 / n_assets, dtype=float)
        else:
            cash_w = max(0.0, 1.0 - float(current.sum()))
            grown = current * (1.0 + ret_arr[i])
            total = float(grown.sum()) + cash_w
            current = grown / total if total > 0 else np.zeros(n_assets, dtype=float)
            if float(current.max()) > 0.495:
                over_mask = current > 0.49
                current[over_mask] = 0.49
        w_arr[i] = current

    return pd.DataFrame(w_arr, index=prices.index, columns=prices.columns)
