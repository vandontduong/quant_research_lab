"""Batch 3 / Cycle 18 (inspired by TradingAgents Multi-Agent Debate, arXiv:2412.20138).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (deterministic rules-based
analogue with no LLMs or agent prompts; Xiao et al., 2024):
- Signal 1 (Trend score): 63d & 126d vol-scaled momentum.
- Signal 2 (Mean-reversion score): opposing 21d standardized return z-score.
- Signal 3 (Macro score): penalizes `SPY`/`TLT` when trailing correlation is positive and
  favors `SHY`/`GLD` when cross-asset volatility spikes.
- Consensus & disagreement dampening: combines the 3 rule scores and shrinks tilts when
  inter-signal variance is high, keeping inverse-volatility risk budgeting as the anchor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v018_trading_agents_debate"
STRATEGY_NOTE = "TradingAgents-inspired (arXiv:2412.20138): 3-signal rules consensus + disagreement dampening (no LLMs)."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "debate_tilt": 0.30,
    "disagreement_veto": 0.70,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "debate_tilt": 0.15},
    {**DEFAULT_PARAMS, "debate_tilt": 0.45},
    {**DEFAULT_PARAMS, "disagreement_veto": 0.40},
    {**DEFAULT_PARAMS, "disagreement_veto": 1.00},
]
HYPOTHESIS = (
    "TradingAgents (arXiv:2412.20138): reconciling opposing Trend Analyst, Mean-Reversion "
    "Researcher, and Macro Strategist scores through a Risk Manager that dampens tilts whenever "
    "inter-agent disagreement is high should filter out noisy trades and improve OOS Sharpe."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal 4-agent debate consensus target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    debate_tilt = float(cfg["debate_tilt"])
    veto_strength = float(cfg["disagreement_veto"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    cols = list(prices.columns)
    spy_idx = cols.index("SPY") if "SPY" in cols else -1
    tlt_idx = cols.index("TLT") if "TLT" in cols else -1
    shy_idx = cols.index("SHY") if "SHY" in cols else -1
    gld_idx = cols.index("GLD") if "GLD" in cols else -1

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    r21 = np.nan_to_num(prices.pct_change(21).to_numpy(dtype=float), nan=0.0)
    r63 = np.nan_to_num(prices.pct_change(63).to_numpy(dtype=float), nan=0.0)
    r126 = np.nan_to_num(prices.pct_change(126).to_numpy(dtype=float), nan=0.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(vol_window, 126)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)

                # Agent 1 — Trend Analyst (bull/bear momentum thesis)
                z63 = r63[i] / (floored * np.sqrt(63.0 / 252.0))
                z126 = r126[i] / (floored * np.sqrt(126.0 / 252.0))
                a_trend = np.tanh(0.5 * z63 + 0.5 * z126)

                # Agent 2 — Mean-Reversion Researcher (opposing contrarian thesis on 21d extremes)
                z21 = r21[i] / (floored * np.sqrt(21.0 / 252.0))
                a_mr = -np.tanh(0.75 * z21)

                # Agent 3 — Macro Strategist (stock-bond correlation & vol-regime thesis)
                a_macro = np.zeros(n_assets, dtype=float)
                if spy_idx >= 0 and tlt_idx >= 0:
                    c_win = ret_arr[i - vol_window + 1 : i + 1]
                    s_std = float(np.std(c_win[:, spy_idx], ddof=1))
                    t_std = float(np.std(c_win[:, tlt_idx], ddof=1))
                    if s_std > 1e-10 and t_std > 1e-10:
                        rho = float(np.corrcoef(c_win[:, spy_idx], c_win[:, tlt_idx])[0, 1])
                        if np.isfinite(rho) and rho > 0.0:
                            a_macro[tlt_idx] = -min(1.0, rho)
                            if shy_idx >= 0:
                                a_macro[shy_idx] = 0.5 * min(1.0, rho)
                            if gld_idx >= 0:
                                a_macro[gld_idx] = 0.5 * min(1.0, rho)

                # Agent 4 — Risk Manager (debate reconciliation + disagreement veto)
                proposals = np.vstack([a_trend, a_mr, a_macro])  # (3, n_assets)
                consensus = proposals.mean(axis=0)
                disagreement = proposals.std(axis=0)
                confidence = np.exp(-veto_strength * disagreement)

                raw = (1.0 / floored) * np.exp(debate_tilt * confidence * consensus)
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
