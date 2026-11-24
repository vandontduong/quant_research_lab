"""Batch 4 (inspired by FinRL Gym-Style Offline Market Environment, arXiv:2011.09607).

Zero-dependency, FinRL-inspired Gym-style offline market environment (`QuantLabEnv`) — not
the FinRL library itself and without Stable-Baselines3/Gymnasium dependencies. Implements
`reset()` and `step(action_weights)` over the frozen ETF price matrix with:
- Strict 1-bar execution lag and realistic weight-drift turnover accounting
- Policy-enforced long-only / gross-leverage / single-asset weight caps
- Configurable reward shaping options:
  * `net_return` (raw net simple return)
  * `vol_scaled` (inspired by arXiv:1911.10107)
  * `asym_inventory` (inspired by arXiv:1804.04216)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from engine.policy import RiskPolicy


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    info: dict[str, Any]


class QuantLabEnv:
    """FinRL-inspired offline Gym-style environment backed by quant_research_lab's RiskPolicy."""

    def __init__(
        self,
        prices: pd.DataFrame,
        policy: RiskPolicy,
        *,
        lookback: int = 21,
        reward_mode: str = "net_return",
        inventory_aversion: float = 0.25,
    ) -> None:
        if len(prices) <= lookback + 2:
            raise ValueError("Price series too short for lookback window.")
        if reward_mode not in {"net_return", "vol_scaled", "asym_inventory"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        self.prices = prices.copy()
        self.policy = policy
        self.lookback = int(lookback)
        self.reward_mode = reward_mode
        self.inventory_aversion = float(inventory_aversion)

        self._rets = self.prices.pct_change().fillna(0.0).to_numpy(dtype=float)
        self._n_days, self._n_assets = self._rets.shape
        exec_cfg = policy.execution
        base_bps = float(exec_cfg["commission_bps_per_side"]) + float(exec_cfg["slippage_bps_per_side"])
        self._cost_rate = base_bps / 10_000.0
        self._max_gross = float(exec_cfg.get("max_gross_leverage", 1.0))
        self._max_single = float(exec_cfg.get("max_single_asset_weight", 0.5))

        self._t = self.lookback
        self._prev_drifted = np.zeros(self._n_assets, dtype=float)
        self._wealth = 1.0
        self._recent_port_rets: list[float] = []

    def reset(self) -> np.ndarray:
        """Reset environment to the first valid causal bar and return observation window."""
        self._t = self.lookback
        self._prev_drifted = np.zeros(self._n_assets, dtype=float)
        self._wealth = 1.0
        self._recent_port_rets = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return strictly causal trailing return window (lookback, n_assets)."""
        return self._rets[self._t - self.lookback + 1 : self._t + 1].copy()

    def _validate_action(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.shape != (self._n_assets,):
            raise ValueError(f"Expected action shape ({self._n_assets},), got {w.shape}.")
        if not np.all(np.isfinite(w)):
            raise ValueError("Action weights contain NaN or Inf.")
        if float(w.min()) < -1e-9:
            raise ValueError("Short weights forbidden in long-only QuantLabEnv.")
        if float(np.abs(w).sum()) > self._max_gross + 1e-6:
            raise ValueError("Action exceeds max_gross_leverage.")
        if float(np.abs(w).max()) > self._max_single + 1e-6:
            raise ValueError("Action exceeds max_single_asset_weight.")
        return w

    def step(self, action_weights: np.ndarray) -> StepResult:
        """Apply action decided at close `t` to next-bar returns at `t + 1` (strict 1-bar lag)."""
        if self._t >= self._n_days - 1:
            raise RuntimeError("Episode already terminated; call reset().")

        w_target = self._validate_action(action_weights)
        next_t = self._t + 1
        r_next = self._rets[next_t]

        one_way_to = float(np.abs(w_target - self._prev_drifted).sum()) * 0.5
        gross_r = float(np.dot(w_target, r_next))
        net_r = gross_r - (one_way_to * 2.0 * self._cost_rate)
        self._wealth *= 1.0 + net_r
        self._recent_port_rets.append(net_r)

        # Update drifted weights
        denom = 1.0 + gross_r
        if denom > 0:
            self._prev_drifted = (w_target * (1.0 + r_next)) / denom
        else:
            self._prev_drifted = np.zeros(self._n_assets, dtype=float)

        reward = self._shape_reward(net_r, w_target)
        self._t = next_t
        terminated = self._t >= self._n_days - 1
        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            terminated=terminated,
            info={
                "date": self.prices.index[self._t].strftime("%Y-%m-%d"),
                "net_return": net_r,
                "turnover": one_way_to,
                "wealth": self._wealth,
            },
        )

    def _shape_reward(self, net_r: float, w_target: np.ndarray) -> float:
        if self.reward_mode == "net_return":
            return net_r
        if self.reward_mode == "vol_scaled":
            window = self._recent_port_rets[-self.lookback :]
            vol = float(np.std(window, ddof=1)) if len(window) >= 5 else 0.01
            return net_r / max(vol, 0.002)
        # asym_inventory (Paper #1): dampens positive speculative gains & penalizes concentration deviation
        equal_w = np.full(self._n_assets, 1.0 / self._n_assets, dtype=float)
        inv_risk = float(np.sum((w_target - equal_w) ** 2))
        dampened_pnl = min(net_r, 0.5 * net_r) if net_r > 0 else net_r
        return dampened_pnl - self.inventory_aversion * inv_risk * 0.01
