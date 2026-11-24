"""Tests proving harness robustness upgrades, DSR multiple-testing math, attribution, and reasoning traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import (
    compute_deflated_sharpe,
    compute_portfolio_attribution,
    compute_probabilistic_sharpe,
)
from engine.data_loader import load_frozen_prices
from engine.policy import load_policy
from engine.reasoning import format_reasoning_trace_markdown, generate_reasoning_trace
from engine.validation import assert_no_lookahead, evaluate_walk_forward
from strategies.v001_inv_vol import DEFAULT_PARAMS as V001_PARAMS
from strategies.v001_inv_vol import generate_weights as v001_weights

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_prices(periods: int = 1200, start: str = "2005-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(start=start, periods=periods)
    cols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"]
    data = {}
    for idx, col in enumerate(cols):
        rets = rng.normal(0.0003 + idx * 0.00003, 0.01, size=periods)
        data[col] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(data, index=dates)


def test_catches_nondeterministic_strategy():
    prices = _synthetic_prices(periods=400)
    counter = {"calls": 0}

    def nondet_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        counter["calls"] += 1
        w = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        w["SPY"] = 0.2 if counter["calls"] % 2 == 1 else 0.4
        return w

    with pytest.raises(AssertionError, match="NONDETERMINISM DETECTED"):
        assert_no_lookahead(nondet_strategy, prices, {})


def test_catches_inplace_prices_and_params_mutation():
    prices = _synthetic_prices(periods=400)

    def mutates_prices(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df.iloc[0, 0] = 9999.0
        return pd.DataFrame(1.0 / len(df.columns), index=df.index, columns=df.columns)

    with pytest.raises(AssertionError, match="INPUT MUTATION DETECTED"):
        assert_no_lookahead(mutates_prices, prices, {})

    def mutates_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        params["leaked"] = True
        return pd.DataFrame(1.0 / len(df.columns), index=df.index, columns=df.columns)

    with pytest.raises(AssertionError, match="INPUT MUTATION DETECTED"):
        assert_no_lookahead(mutates_params, prices, {"window": 20})


def test_catches_fixed_length_future_shock_lookahead():
    prices = _synthetic_prices(periods=600)
    full_len = len(prices)

    # Adversarial strategy that only leaks when given the full-length DataFrame
    # (evading a naive truncation-only check on shorter slices if it used a length guard,
    #  or using full-sample global normalization on equal-length shocked inputs).
    def length_guarded_leaker(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        w = pd.DataFrame(1.0 / len(df.columns), index=df.index, columns=df.columns)
        if len(df) == full_len:
            # Uses final-row SPY price to scale early weights (global lookahead)
            scale = float(df["SPY"].iloc[-1]) / float(df["SPY"].iloc[0])
            w["SPY"] = min(0.45, 0.1 * scale)
        return w

    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        assert_no_lookahead(length_guarded_leaker, prices, {}, probe_offsets=(260,))


def test_data_loader_rejects_duplicate_dates_and_nonpositive_prices(tmp_path):
    csv_path = tmp_path / "etf_daily_total_return.csv"
    manifest_path = tmp_path / "manifest.json"

    # 1) Duplicate dates
    csv_dup = "date,SPY,SHY\n2020-01-02,100.0,85.0\n2020-01-02,101.0,85.1\n"
    csv_path.write_text(csv_dup, encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        load_frozen_prices(tmp_path, verify_hash=True)

    # 2) Non-positive price
    csv_zero = "date,SPY,SHY\n2020-01-02,100.0,85.0\n2020-01-03,0.0,85.1\n"
    csv_path.write_text(csv_zero, encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-positive"):
        load_frozen_prices(tmp_path, verify_hash=True)


def test_deflated_sharpe_penalizes_multiple_testing():
    psr_small = compute_probabilistic_sharpe(0.8, observations=252)
    psr_large = compute_probabilistic_sharpe(0.8, observations=2520)
    assert psr_large > psr_small > 0.5

    dsr_1 = compute_deflated_sharpe(0.9, observations=1500, trial_count=1)
    dsr_10 = compute_deflated_sharpe(0.9, observations=1500, trial_count=10)
    dsr_100 = compute_deflated_sharpe(0.9, observations=1500, trial_count=100)
    assert dsr_1 > dsr_10 > dsr_100


def test_portfolio_attribution_accuracy_and_walk_forward_diagnostics():
    dates = pd.bdate_range("2020-01-02", periods=60)
    # Deterministic upward SPY, flat SHY
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * (1.001 ** np.arange(60)),
            "SHY": np.full(60, 100.0),
        },
        index=dates,
    )
    eff_w = pd.DataFrame({"SPY": 0.5, "SHY": 0.0}, index=dates)
    net_r = (prices["SPY"].pct_change().fillna(0.0) * 0.5).rename("net_return")

    attrib = compute_portfolio_attribution(prices, eff_w, net_r, benchmark_ticker="SPY")
    assert attrib["avg_weights"]["SPY"] == pytest.approx(0.5, abs=1e-4)
    assert attrib["avg_cash_weight"] == pytest.approx(0.5, abs=1e-4)
    assert attrib["avg_effective_bets"] == pytest.approx(1.0, abs=1e-3)
    assert attrib["top_contributor"]["ticker"] == "SPY"
    assert attrib["top_contributor"]["annualized_return"] > 0.10

    # End-to-end walk-forward diagnostics on synthetic multi-year data
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    synth = _synthetic_prices(periods=2500, start="2005-01-03")
    res = evaluate_walk_forward(v001_weights, synth, V001_PARAMS, policy)
    assert "attribution" in res
    assert "fold_consistency" in res
    fc = res["fold_consistency"]
    assert 0.0 <= fc["positive_fold_ratio"] <= 1.0
    assert fc["min_fold_sharpe"] <= fc["median_fold_sharpe"] <= fc["max_fold_sharpe"]


def test_concise_reasoning_traces_diagnose_distinct_outcomes():
    champ = {
        "strategy_name": "v001_inv_vol",
        "out_of_sample": {"sharpe": 1.038, "max_drawdown": -0.161, "annual_turnover": 0.39},
    }

    # 1) Drawdown breach discard
    dd_fail = {
        "cycle_id": "cycle_003",
        "trial_count": 4,
        "strategy_name": "v003_momentum",
        "in_sample": {"sharpe": 0.90},
        "out_of_sample": {"sharpe": 0.69, "max_drawdown": -0.274, "annual_turnover": 2.61},
        "deflated_sharpe": 0.72,
        "gates": {"promote": False, "failed_gates": ["max_oos_drawdown", "beats_champion_margin"]},
        "decision": "DISCARD",
    }
    t_dd = generate_reasoning_trace(dd_fail, prior_champion=champ)
    assert len(t_dd["summary_lines"]) == 5
    assert "max_oos_drawdown" in t_dd["verdict"]
    assert "volatility scaling" in t_dd["next_action_hint"].lower()

    # 2) Sub-margin near-miss discard
    near_miss = {
        "cycle_id": "cycle_006",
        "trial_count": 7,
        "strategy_name": "v006_vol_target",
        "in_sample": {"sharpe": 0.87},
        "out_of_sample": {"sharpe": 1.055, "max_drawdown": -0.182, "annual_turnover": 1.49},
        "deflated_sharpe": 0.94,
        "gates": {"promote": False, "failed_gates": ["beats_champion_margin"]},
        "decision": "DISCARD",
    }
    t_nm = generate_reasoning_trace(near_miss, prior_champion=champ)
    assert "sub-margin" in t_nm["next_action_hint"].lower()
    md = format_reasoning_trace_markdown({**near_miss, "reasoning_trace": t_nm})
    assert "### `cycle_006`" in md

    # 3) Crash trace
    crash_rec = {
        "cycle_id": "cycle_007",
        "strategy_name": "v007_broken",
        "decision": "CRASH",
        "error": "AssertionError: LOOKAHEAD DETECTED",
    }
    t_cr = generate_reasoning_trace(crash_rec, prior_champion=champ)
    assert len(t_cr["summary_lines"]) == 2
    assert "CRASH" in t_cr["verdict"]
