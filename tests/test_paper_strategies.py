"""Algorithmic trading rigor tests for stylized paper-inspired strategies (v010–v019) and QuantLabEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import validate_weights
from engine.policy import load_policy
from engine.validation import assert_no_lookahead
from strategies import (
    v010_dmn_sharpe_opt,
    v011_momentum_transformer_cpd,
    v012_drl_vol_scaled_policy,
    v013_sdf_macro_factor,
    v014_stat_arb_residual_mr,
    v015_eiie_portfolio_rl,
    v016_mm_deeplob_inventory,
    v017_news_underreaction_pead,
    v018_trading_agents_debate,
    v019_qlib_factor_ensemble,
)
from tools.finrl_env import QuantLabEnv

ROOT = Path(__file__).resolve().parents[1]

PAPER_STRATEGIES = [
    v010_dmn_sharpe_opt,
    v011_momentum_transformer_cpd,
    v012_drl_vol_scaled_policy,
    v013_sdf_macro_factor,
    v014_stat_arb_residual_mr,
    v015_eiie_portfolio_rl,
    v016_mm_deeplob_inventory,
    v017_news_underreaction_pead,
    v018_trading_agents_debate,
    v019_qlib_factor_ensemble,
]


def _synthetic_prices(periods: int = 700, start: str = "2010-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range(start=start, periods=periods)
    cols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"]
    vols = [0.011, 0.013, 0.014, 0.009, 0.010, 0.0015]
    data = {}
    for col, vol in zip(cols, vols, strict=True):
        rets = rng.normal(0.00025, vol, size=periods)
        data[col] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(data, index=dates)


@pytest.mark.parametrize("mod", PAPER_STRATEGIES, ids=lambda m: m.STRATEGY_NAME)
def test_paper_strategy_is_causal_deterministic_immutable_and_valid(mod):
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_prices(periods=650)

    # Full adversarial check: determinism, input immutability, prefix-truncation & future-shock invariance
    assert_no_lookahead(mod.generate_weights, prices, mod.DEFAULT_PARAMS, probe_offsets=(300, 520))

    # Weight constraints: finite, long-only, gross leverage <= 1.0, single-asset <= 0.50
    w = mod.generate_weights(prices, mod.DEFAULT_PARAMS)
    validate_weights(w, policy)


def test_finrl_gym_env_lag_costs_and_reward_modes():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_prices(periods=120)
    n_assets = prices.shape[1]
    equal_w = np.full(n_assets, 1.0 / n_assets, dtype=float)

    for mode in ("net_return", "vol_scaled", "asym_inventory"):
        env = QuantLabEnv(prices, policy, lookback=21, reward_mode=mode)
        obs = env.reset()
        assert obs.shape == (21, n_assets)

        steps = 0
        terminated = False
        while not terminated:
            res = env.step(equal_w)
            assert np.isfinite(res.reward)
            assert res.observation.shape == (21, n_assets)
            terminated = res.terminated
            steps += 1
        assert steps == len(prices) - 21 - 1
        assert res.info["wealth"] > 0.0

    # Policy violations must be rejected immediately by QuantLabEnv
    env_guard = QuantLabEnv(prices, policy, lookback=21)
    env_guard.reset()
    bad_short = np.array([-0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="Short weights forbidden"):
        env_guard.step(bad_short)

    bad_conc = np.array([0.8, 0.04, 0.04, 0.04, 0.04, 0.04])
    with pytest.raises(ValueError, match="max_single_asset_weight"):
        env_guard.step(bad_conc)
