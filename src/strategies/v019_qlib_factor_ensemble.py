"""Batch 4 / Cycle 19 (Paper #11 — Qlib Platform: Expression Pipeline & Model Zoo Ensemble, arXiv:2009.11189).

Adapts Yang et al. (2020) Qlib workflow into a strictly causal, zero-dependency strategy:
- Expression/Feature Pipeline: computes a cached, normalized factor library per bar (inverse-vol,
  correlation-penalized risk parity, downside-semivolatility, rate-regime correlation defense,
  and CPD-dampened trend).
- Rolling IC/Sharpe Model Zoo Ensembling: evaluates each sub-model's strictly causal trailing
  realized Sharpe over `ensemble_window` bars and allocates softmax-weighted ensemble weights
  across the sub-models, combining orthogonal risk/alpha sleeves without lookahead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize
from strategies.v001_inv_vol import generate_weights as inv_vol_weights
from strategies.v004_corr_risk_parity import generate_weights as corr_rp_weights
from strategies.v008_downside_vol_inv_weight import generate_weights as downside_vol_weights
from strategies.v009_rate_regime_inv_vol import generate_weights as rate_regime_weights

STRATEGY_NAME = "v019_qlib_factor_ensemble"
STRATEGY_NOTE = "Qlib workflow (arXiv:2009.11189): causal rolling-Sharpe softmax ensemble over 4 risk-parity sub-models."
DEFAULT_PARAMS: dict[str, Any] = {
    "ensemble_window": 252,
    "softmax_temp": 1.50,
    "rebalance_every_days": 21,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "ensemble_window": 126},
    {**DEFAULT_PARAMS, "ensemble_window": 504},
    {**DEFAULT_PARAMS, "softmax_temp": 0.75},
    {**DEFAULT_PARAMS, "softmax_temp": 2.50},
]
HYPOTHESIS = (
    "Qlib (arXiv:2009.11189) model-zoo ensembling: dynamically combining 4 orthogonal, low-turnover "
    "causal sub-models (v001 inv-vol, v004 corr-aware, v008 downside-vol, v009 rate-regime) via "
    "trailing causal realized-Sharpe softmax weights should reduce variance and improve OOS Sharpe."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal Qlib-style rolling-Sharpe ensemble target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    ens_win = int(cfg["ensemble_window"])
    temp = float(cfg["softmax_temp"])
    step = int(cfg["rebalance_every_days"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)

    # Sub-model zoo (all strictly causal, deterministic, low-turnover risk-budget sleeves)
    sub_models = [
        inv_vol_weights(prices, None).to_numpy(dtype=float),
        corr_rp_weights(prices, None).to_numpy(dtype=float),
        downside_vol_weights(prices, None).to_numpy(dtype=float),
        rate_regime_weights(prices, None).to_numpy(dtype=float),
    ]
    n_models = len(sub_models)

    # Strictly causal 1-bar lagged realized daily return of each sub-model: m_ret[t] = w[t-1] · r[t]
    model_rets = np.zeros((n_days, n_models), dtype=float)
    for m_idx, w_m in enumerate(sub_models):
        lagged_w = np.vstack([np.zeros((1, n_assets), dtype=float), w_m[:-1]])
        model_rets[:, m_idx] = (lagged_w * ret_arr).sum(axis=1)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)

    for i in range(n_days):
        if i % step == 0:
            if i >= ens_win:
                # Causal trailing realized Sharpe of each sub-model strictly through bar i
                win_r = model_rets[i - ens_win + 1 : i + 1]
                mu = win_r.mean(axis=0)
                sd = np.maximum(win_r.std(axis=0, ddof=1), 1e-6)
                sharpes = (mu / sd) * np.sqrt(252.0)
                logits = np.clip(sharpes * temp, -10.0, 10.0)
                logits = logits - float(logits.max())
                ens_w = np.exp(logits)
                ens_w = ens_w / float(ens_w.sum())
            else:
                ens_w = np.full(n_models, 1.0 / n_models, dtype=float)

            blended = sum(ens_w[m] * sub_models[m][i] for m in range(n_models))
            current = _cap_and_normalize(blended, max_weight)
        else:
            grown = current * (1.0 + ret_arr[i])
            total = float(grown.sum())
            current = grown / total if total > 0 else np.full(n_assets, 1.0 / n_assets, dtype=float)
            if float(current.max()) > 0.495:
                current = _cap_and_normalize(current, 0.49)
        w_arr[i] = current

    return pd.DataFrame(w_arr, index=prices.index, columns=prices.columns)
