"""Batch 3 / Cycle 17 (inspired by LLM News Sentiment / Underreaction Drift, arXiv:2304.07619).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (no news text, no headlines,
and no LLM API calls; Lopez-Lira & Tang, 2023):
- Price-shock proxy: flags large causal 3-day standardized return moves (`|z_3d| > shock_threshold`)
  as +1 / 0 / -1 event proxies in place of LLM headline sentiment scores.
- Exponentially decays those return-shock indicators (`sentiment_decay`) to test post-shock
  drift tilts on top of inverse-volatility weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v017_news_underreaction_pead"
STRATEGY_NOTE = "News-drift proxy (arXiv:2304.07619): decaying 3d return-shock proxy on inv-vol (no news text/LLM)."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "shock_threshold": 1.50,
    "sentiment_decay": 0.92,
    "sentiment_tilt": 0.25,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "shock_threshold": 1.20},
    {**DEFAULT_PARAMS, "shock_threshold": 1.80},
    {**DEFAULT_PARAMS, "sentiment_decay": 0.88},
    {**DEFAULT_PARAMS, "sentiment_decay": 0.96},
]
HYPOTHESIS = (
    "LLM News Sentiment Trading (arXiv:2304.07619): an offline causal proxy scoring high-salience "
    "return shocks (|z_3d| > 1.5) as +1/0/-1 sentiment events with exponential decay should capture "
    "market underreaction drift on top of inverse-volatility weighting."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal post-event underreaction drift weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    shock_thresh = float(cfg["shock_threshold"])
    decay = float(cfg["sentiment_decay"])
    sent_tilt = float(cfg["sentiment_tilt"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    r3 = np.nan_to_num(prices.pct_change(3).to_numpy(dtype=float), nan=0.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    sentiment_state = np.zeros(n_assets, dtype=float)

    for i in range(n_days):
        vols = rolling_vol[i]
        if i >= vol_window and np.all(np.isfinite(vols)):
            floored = np.maximum(vols, min_vol_floor)
            z3 = r3[i] / (floored * np.sqrt(3.0 / 252.0))
            # Discrete +1 / 0 / -1 event sentiment score (mirroring good/neutral/bad LLM headline scoring)
            event_score = np.where(z3 > shock_thresh, 1.0, np.where(z3 < -shock_thresh, -1.0, 0.0))
            sentiment_state = decay * sentiment_state + (1.0 - decay) * event_score

        if i % step == 0:
            if i >= vol_window and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                clipped_sent = np.clip(sentiment_state * 3.0, -1.5, 1.5)
                raw = (1.0 / floored) * np.exp(sent_tilt * clipped_sent)
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
