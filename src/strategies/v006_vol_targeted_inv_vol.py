"""Cycle 6 Candidate: Volatility-managed inverse-volatility portfolio (Moreira-Muir style).

Strictly causal: on each scheduled rebalance bar `t`, constructs inverse-volatility
weights across the 5 risky assets (SPY, QQQ, IWM, TLT, GLD), estimates their trailing
`fast_vol_window`-day realized portfolio volatility `sigma_p`, scales total risky
exposure by `min(1.0, target_vol / sigma_p)` (capped at 1.0 - min_defensive_weight),
and allocates the remainder to defensive SHY (up to `max_weight` = 0.45, excess in cash).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v006_vol_targeted_inv_vol"
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "fast_vol_window": 21,
    "target_vol": 0.08,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
    "defensive_ticker": "SHY",
}
HYPOTHESIS = (
    "Volatility clusters more reliably than returns: scale v001 risky-sleeve exposure "
    "inversely to its fast 21-day realized portfolio volatility (`min(1.0, 8% / sigma_p)`), "
    "routing excess weight into SHY (capped at 45%) and cash during crisis volatility spikes."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal volatility-targeted inverse-volatility target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    fast_window = int(cfg["fast_vol_window"])
    target_vol = float(cfg["target_vol"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])
    defensive = str(cfg["defensive_ticker"]).upper()

    cols = list(prices.columns)
    def_idx = cols.index(defensive) if defensive in cols else -1
    risky_idx = [j for j in range(len(cols)) if j != def_idx]

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    w_arr = np.zeros((n_days, n_assets), dtype=float)

    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, fast_window)

    for i in range(n_days):
        if i % step == 0:
            if i >= min_bars:
                slow_r = ret_arr[i - vol_window + 1 : i + 1, risky_idx]
                risky_vols = np.std(slow_r, axis=0, ddof=1) * np.sqrt(252.0)
                if np.all(np.isfinite(risky_vols)):
                    floored = np.maximum(risky_vols, min_vol_floor)
                    risky_w = _cap_and_normalize(1.0 / floored, max_weight)

                    # Realized portfolio volatility of the risky sleeve over fast_window.
                    fast_r = ret_arr[i - fast_window + 1 : i + 1, risky_idx]
                    port_daily = fast_r @ risky_w
                    port_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252.0))
                    scale = min(1.0, target_vol / max(port_vol, 1e-6))

                    w_full = np.zeros(n_assets, dtype=float)
                    for pos, asset_j in enumerate(risky_idx):
                        w_full[asset_j] = risky_w[pos] * scale
                    if def_idx >= 0:
                        rem = max(0.0, 1.0 - float(w_full.sum()))
                        w_full[def_idx] = min(rem, max_weight)
                    current = w_full
                else:
                    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
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
