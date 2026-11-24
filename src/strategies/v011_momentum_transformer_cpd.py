"""Batch 1 / Cycle 11 (Paper #3 — Momentum Transformer + CPD, arXiv:2112.08534).

Adapts Wood, Giegerich, Roberts & Zohren (2021) to long-only daily ETFs:
- Multi-timescale attention layer: computes softmax attention weights over 4 trend horizons
  (21d, 63d, 126d, 252d) based on each horizon's trailing signal-to-noise (trend persistence),
  preserving long-horizon context to avoid fast-reversion whipsaws.
- Causal online Changepoint Detection (CPD): measures a standardized mean-shift severity
  statistic comparing recent 21d return mean vs. prior 105d baseline (scaled by trailing vol).
  When CPD severity spikes (regime break / choppy reversal), smoothly dampens the trend tilt
  back toward defensive inverse-volatility weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v011_momentum_transformer_cpd"
STRATEGY_NOTE = "Momentum Transformer + CPD (arXiv:2112.08534): attention over 4 trend horizons dampened by causal changepoint severity."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "cpd_short": 21,
    "cpd_long": 126,
    "tilt_strength": 0.35,
    "cpd_dampening": 0.80,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "tilt_strength": 0.20},
    {**DEFAULT_PARAMS, "tilt_strength": 0.50},
    {**DEFAULT_PARAMS, "cpd_dampening": 0.40},
    {**DEFAULT_PARAMS, "cpd_dampening": 1.20},
]
HYPOTHESIS = (
    "Momentum Transformer + CPD (arXiv:2112.08534): softmax attention across 21d/63d/126d/252d "
    "trend horizons preserves long-term context, while an online changepoint-severity gate "
    "dampens trend tilts during regime breaks to prevent whipsaw turnover."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal attention-weighted momentum + CPD-dampened inverse-vol weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    cpd_short = int(cfg["cpd_short"])
    cpd_long = int(cfg["cpd_long"])
    tilt_strength = float(cfg["tilt_strength"])
    cpd_damp = float(cfg["cpd_dampening"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )

    horizons = (21, 63, 126, 252)
    mom_stack = np.stack(
        [np.nan_to_num(prices.pct_change(h).to_numpy(dtype=float), nan=0.0) for h in horizons],
        axis=-1,
    )  # (n_days, n_assets, 4)
    h_scales = np.sqrt(np.array(horizons, dtype=float) / 252.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(252, cpd_long, vol_window)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                # Multi-timescale vol-normalized trend scores: (n_assets, 4)
                z_h = mom_stack[i] / (floored[:, None] * h_scales[None, :])
                z_clipped = np.clip(z_h, -2.0, 2.0)

                # Attention weights favor horizons with clearer directional persistence (higher |z|, biased longer)
                prior_bias = np.array([0.0, 0.15, 0.30, 0.45], dtype=float)
                logits = np.abs(z_clipped) + prior_bias[None, :]
                logits = logits - logits.max(axis=1, keepdims=True)
                attn = np.exp(logits)
                attn = attn / np.maximum(attn.sum(axis=1, keepdims=True), 1e-12)
                attn_trend = (attn * z_clipped).sum(axis=1)

                # Causal changepoint severity: standardized shift between recent cpd_short and prior baseline
                short_r = ret_arr[i - cpd_short + 1 : i + 1]
                base_r = ret_arr[i - cpd_long + 1 : i - cpd_short + 1]
                mean_diff = np.abs(short_r.mean(axis=0) - base_r.mean(axis=0))
                se = np.maximum(np.std(base_r, axis=0, ddof=1), 1e-6) / np.sqrt(cpd_short)
                cpd_severity = np.clip(mean_diff / se, 0.0, 4.0)
                # Gate in (0, 1]: approaches 0 when regime break severity is high
                cpd_gate = np.exp(-cpd_damp * np.maximum(cpd_severity - 1.0, 0.0))

                effective_tilt = np.exp(tilt_strength * cpd_gate * np.tanh(attn_trend))
                raw = (1.0 / floored) * effective_tilt
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
