"""Batch 1 / Cycle 12 (inspired by Deep RL for Trading, arXiv:1911.10107).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (not a deep RL reproduction
of Zhang, Zohren & Roberts, 2019):
- Lightweight online linear policy update (3-feature shared weight vector `theta`, no neural
  network or replay buffer) trained on volatility-scaled, cost-penalized returns:
      R_t = (w_{t-1} · r_t) / sigma_{t-1} - cost_penalty * turnover_t
- State features per asset: trailing 21d & 63d vol-scaled returns and trailing volatility ratio
  (21d vol / 63d vol).
- Updates `theta` strictly causally after each realized rebalance period using reward minus
  an exponentially weighted baseline with L2 decay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v012_drl_vol_scaled_policy"
STRATEGY_NOTE = "DRL Trading-inspired (arXiv:1911.10107): 3-feature online linear policy update (no deep RL)."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "fast_vol_window": 21,
    "learning_rate": 0.05,
    "cost_penalty": 0.001,
    "rebalance_every_days": 21,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "learning_rate": 0.02},
    {**DEFAULT_PARAMS, "learning_rate": 0.10},
    {**DEFAULT_PARAMS, "cost_penalty": 0.0005},
    {**DEFAULT_PARAMS, "cost_penalty": 0.002},
]
HYPOTHESIS = (
    "Deep RL for Trading (arXiv:1911.10107): a causal online policy-gradient agent trained "
    "on volatility-scaled, transaction-cost-penalized realized rewards should adaptively weight "
    "momentum and vol-regime features without lookahead or excessive turnover."
)


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal online policy-gradient weights trained on vol-scaled net rewards."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    fast_window = int(cfg["fast_vol_window"])
    lr = float(cfg["learning_rate"])
    cost_pen = float(cfg["cost_penalty"])
    step = int(cfg["rebalance_every_days"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    slow_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    fast_vol = (rets.rolling(fast_window, min_periods=fast_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    mom21 = np.nan_to_num(prices.pct_change(21).to_numpy(dtype=float), nan=0.0)
    mom63 = np.nan_to_num(prices.pct_change(63).to_numpy(dtype=float), nan=0.0)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)

    # Shared policy parameters over 3 causal state features: [mom21_z, mom63_z, -vol_ratio_z]
    theta = np.array([0.05, 0.10, -0.10], dtype=float)
    baseline_reward = 0.0
    last_rebal_idx: int | None = None
    last_state: np.ndarray | None = None
    last_action_w: np.ndarray | None = None
    last_turnover = 0.0

    min_bars = max(vol_window, fast_window, 63)

    for i in range(n_days):
        if i % step == 0:
            vols = slow_vol[i]
            fvols = fast_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)) and np.all(np.isfinite(fvols)):
                floored = np.maximum(vols, min_vol_floor)

                # 1) Causal policy update using realized reward over (last_rebal_idx, i]
                if last_rebal_idx is not None and last_state is not None and last_action_w is not None:
                    period_r = ret_arr[last_rebal_idx + 1 : i + 1]
                    if len(period_r) > 0:
                        port_daily = period_r @ last_action_w
                        realized_ret = float(np.prod(1.0 + port_daily) - 1.0)
                        realized_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252.0)) if len(port_daily) > 1 else 0.10
                        vol_scaled_reward = realized_ret / max(realized_vol, 0.04) - cost_pen * last_turnover
                        advantage = np.clip(vol_scaled_reward - baseline_reward, -2.0, 2.0)
                        baseline_reward = 0.90 * baseline_reward + 0.10 * vol_scaled_reward

                        # Softmax policy gradient: d log pi / d theta = sum_j w_j * (s_j - E_w[s])
                        exp_s = last_action_w @ last_state  # (3,)
                        # Weight update nudged toward features of assets that outperformed
                        asset_cum = np.prod(1.0 + period_r, axis=0) - 1.0
                        asset_adv = (asset_cum - float(asset_cum.mean())) / max(float(asset_cum.std()) + 1e-6, 0.01)
                        grad = (last_action_w * asset_adv) @ (last_state - exp_s[None, :])
                        theta = np.clip(0.98 * theta + lr * advantage * grad, -1.0, 1.0)

                # 2) Construct current state (n_assets, 3) and sample deterministic actor policy
                m21_z = np.tanh(mom21[i] / (floored * np.sqrt(21.0 / 252.0)))
                m63_z = np.tanh(mom63[i] / (floored * np.sqrt(63.0 / 252.0)))
                v_ratio = np.clip((fvols / floored) - 1.0, -1.0, 1.0)
                state = np.stack([m21_z, m63_z, v_ratio], axis=-1)

                logits = state @ theta
                raw = (1.0 / floored) * np.exp(0.35 * np.tanh(logits))
                new_w = _cap_and_normalize(raw, max_weight)

                last_turnover = float(np.abs(new_w - current).sum()) * 0.5
                last_rebal_idx = i
                last_state = state
                last_action_w = new_w.copy()
                current = new_w
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
