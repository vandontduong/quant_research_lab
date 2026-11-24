"""Offline unit tests verifying harness causality, risk ceilings, holdout lock, and cost accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import run_backtest, validate_weights
from engine.policy import evaluate_gates, load_policy
from engine.validation import assert_no_lookahead, evaluate_walk_forward, generate_walk_forward_folds
from strategies.v000_baseline import DEFAULT_PARAMS, generate_weights

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_prices(start: str = "2005-01-03", periods: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=periods)
    cols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"]
    data = {}
    for idx, col in enumerate(cols):
        rets = rng.normal(0.0003 + idx * 0.00002, 0.01, size=periods)
        data[col] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(data, index=dates)


def test_policy_loads_with_sha256_and_blocks_live_execution(tmp_path):
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    assert len(policy.sha256) == 64
    assert policy.raw["research_only"] is True
    assert policy.raw["allow_live_execution"] is False

    bad = dict(policy.raw)
    bad["allow_live_execution"] = True
    bad_path = tmp_path / "bad_policy.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="live execution is forbidden"):
        load_policy(bad_path)


def test_prefix_invariance_passes_for_baseline_and_catches_lookahead():
    prices = _synthetic_prices(periods=1000)
    # Causal baseline must pass
    assert_no_lookahead(generate_weights, prices, DEFAULT_PARAMS)

    # Leaky strategy peeking at tomorrow's return must fail loudly
    def leaky_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        future_ret = df.pct_change().shift(-1).fillna(0.0)
        best = future_ret.eq(future_ret.max(axis=1), axis=0).astype(float)
        return best.div(best.sum(axis=1), axis=0).fillna(0.0)

    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        assert_no_lookahead(leaky_strategy, prices, {})


def test_validate_weights_rejects_nan_leverage_short_and_concentration():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    dates = pd.bdate_range("2020-01-02", periods=5)
    cols = policy.universe

    w_nan = pd.DataFrame(0.1, index=dates, columns=cols)
    w_nan.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_weights(w_nan, policy)

    w_short = pd.DataFrame(0.1, index=dates, columns=cols)
    w_short.iloc[1, 0] = -0.05
    with pytest.raises(ValueError, match="Short weight"):
        validate_weights(w_short, policy)

    w_lev = pd.DataFrame(0.3, index=dates, columns=cols)  # sum = 1.8 > 1.0
    with pytest.raises(ValueError, match="Gross leverage"):
        validate_weights(w_lev, policy)

    w_conc = pd.DataFrame(0.0, index=dates, columns=cols)
    w_conc["SPY"] = 0.60  # > 0.50 cap
    with pytest.raises(ValueError, match="Single-asset weight"):
        validate_weights(w_conc, policy)


def test_holdout_leakage_is_blocked_by_walk_forward():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    leaky_prices = _synthetic_prices(start="2020-01-02", periods=1500)  # runs past 2024-01-02
    with pytest.raises(ValueError, match="Holdout leakage blocked"):
        evaluate_walk_forward(generate_weights, leaky_prices, DEFAULT_PARAMS, policy)


def test_walk_forward_purge_gap_and_transaction_cost_drag():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_prices(start="2005-01-03", periods=2500)
    folds = generate_walk_forward_folds(
        prices.index,
        train_days=1260,
        purge_days=21,
        test_days=504,
        step_days=504,
    )
    assert len(folds) >= 2
    for f in folds:
        assert f["test_start"] - f["train_end"] == 21

    w = generate_weights(prices, {"rebalance_every_days": 5})
    bt = run_backtest(prices, w, policy)
    assert bt["metrics"]["annual_turnover"] > 0
    assert bt["metrics"]["rebalance_events"] > 0

    # Gate checks enforce promotion margin
    gate_fail = evaluate_gates(
        policy,
        is_metrics={"sharpe": 1.0},
        oos_metrics={"sharpe": 0.82, "max_drawdown": -0.15, "annual_turnover": 1.0, "rebalance_events": 40},
        perturbation_drop=0.05,
        champion_oos_sharpe=0.80,
    )
    assert gate_fail["promote"] is False
    assert "beats_champion_margin" in gate_fail["failed_gates"]


def test_meta_diagnostics_tail_risk_regimes_and_cost_monotonicity():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_prices(start="2005-01-03", periods=4600)
    prices = prices.loc[prices.index < pd.Timestamp(policy.splits["holdout_start"])]

    res = evaluate_walk_forward(generate_weights, prices, {"rebalance_every_days": 5}, policy)
    oos = res["out_of_sample"]
    assert oos["worst_day"] <= oos["cvar_95_daily"] <= 0.0

    # Cost sensitivity must be monotonically non-increasing in CAGR as cost multiplier rises (1x -> 2x -> 3x)
    cs = res["cost_sensitivity"]
    assert list(cs.keys()) == ["1x_5bps", "2x_10bps", "3x_15bps"]
    assert cs["1x_5bps"]["oos_cagr"] >= cs["2x_10bps"]["oos_cagr"] >= cs["3x_15bps"]["oos_cagr"]

    # Named stress regimes within development window are populated
    regimes = res["regime_breakdown"]
    assert "2020_covid_crash" in regimes
    assert "2022_rate_shock" in regimes

