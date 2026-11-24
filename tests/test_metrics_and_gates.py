"""Unit tests for PPF/DSR math, Sortino ratio, required perturbation gates, and ledger git provenance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from engine.backtest import _norm_ppf_approx, compute_deflated_sharpe, summarize_returns
from engine.data_loader import load_frozen_prices
from engine.policy import evaluate_gates, load_policy
from engine.runner import run_cycle
from engine.validation import evaluate_perturbation_stability, evaluate_walk_forward
from strategies import v001_inv_vol
from strategies.v000_baseline import generate_weights as v000_weights

ROOT = Path(__file__).resolve().parents[1]


def test_norm_ppf_approx_tail_accuracy_and_symmetry():
    expected = {
        0.90: 1.281552,
        0.95: 1.644854,
        0.99: 2.326348,
        0.999: 3.090232,
        0.10: -1.281552,
        0.05: -1.644854,
        0.01: -2.326348,
    }
    for p, target in expected.items():
        val = _norm_ppf_approx(p)
        assert val == pytest.approx(target, abs=1e-4)

    for p in (0.01, 0.05, 0.10, 0.90, 0.95, 0.99, 0.999):
        assert _norm_ppf_approx(1.0 - p) == pytest.approx(-_norm_ppf_approx(p), abs=1e-12)


def test_compute_deflated_sharpe_realistic_hurdles_and_values():
    euler_gamma = 0.5772156649015329
    sharpe_std = 0.25

    for n_trials in (8, 20):
        z1 = _norm_ppf_approx(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf_approx(1.0 - 1.0 / (n_trials * math.e))
        sr0_annual = sharpe_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
        # Realistic extreme-value hurdle for N=8..20 with sharpe_std_annual=0.25 (~0.35 to ~0.55), not ~2.53.
        assert 0.30 < sr0_annual < 0.70

    dsr_8 = compute_deflated_sharpe(1.0, observations=3500, trial_count=8, sharpe_std_annual=sharpe_std)
    dsr_20 = compute_deflated_sharpe(1.0, observations=3500, trial_count=20, sharpe_std_annual=sharpe_std)
    assert 0.90 < dsr_20 < dsr_8 < 1.0


def test_summarize_returns_sortino_full_sample_downside_deviation():
    # Construct known 5-day return series with positive, zero, and negative days.
    dates = pd.bdate_range("2020-01-02", periods=5)
    rets = pd.Series([0.02, -0.01, 0.00, 0.03, -0.02], index=dates)
    turnover = pd.Series([0.1, 0.0, 0.0, 0.0, 0.0], index=dates)

    # Full-sample downside deviation includes zeros on non-negative days:
    # min(r, 0)^2 = [0, 0.0001, 0, 0, 0.0004], mean = 0.0001, sqrt = 0.01.
    mean_daily = float(rets.mean())  # 0.004
    expected_downside_std = math.sqrt((0.01**2 + 0.02**2) / 5.0)
    expected_sortino = (mean_daily / expected_downside_std) * math.sqrt(252.0)

    metrics = summarize_returns(rets, turnover, rebalance_events=1)
    assert metrics["sortino"] == pytest.approx(round(expected_sortino, 6), abs=1e-6)
    assert metrics["sortino"] > metrics["sharpe"]


def test_v001_sortino_exceeds_sharpe_and_neighbor_perturbation_passes():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    dev_prices, _ = load_frozen_prices(
        ROOT / "data",
        tickers=policy.universe,
        start=policy.splits["development_start"],
        end=policy.splits["development_end"],
        verify_hash=True,
    )
    assert len(v001_inv_vol.NEIGHBOR_PARAMS) >= 4

    res = evaluate_walk_forward(
        v001_inv_vol.generate_weights,
        dev_prices,
        v001_inv_vol.DEFAULT_PARAMS,
        policy,
        include_diagnostics=False,
    )
    oos = res["out_of_sample"]
    assert oos["sortino"] > oos["sharpe"]

    pdrop = evaluate_perturbation_stability(
        v001_inv_vol.generate_weights,
        dev_prices,
        v001_inv_vol.DEFAULT_PARAMS,
        v001_inv_vol.NEIGHBOR_PARAMS,
        policy,
        float(oos["sharpe"]),
    )
    assert pdrop is not None
    assert math.isfinite(pdrop)
    assert pdrop <= float(policy.risk_gates["max_perturbation_sharpe_drop"])


def test_evaluate_gates_fails_closed_when_perturbation_drop_missing():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    is_metrics = {"sharpe": 1.0}
    oos_metrics = {
        "sharpe": 1.10,
        "max_drawdown": -0.10,
        "annual_turnover": 0.50,
        "rebalance_events": 60,
    }

    missing_res = evaluate_gates(
        policy,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        perturbation_drop=None,
        champion_oos_sharpe=0.95,
    )
    assert missing_res["promote"] is False
    assert missing_res["risk_passed"] is False
    assert "perturbation_stability" in missing_res["failed_gates"]
    check = missing_res["checks"]["perturbation_stability"]
    assert check["passed"] is False
    assert check["actual"] is None
    assert check["limit"] == float(policy.risk_gates["max_perturbation_sharpe_drop"])

    passing_res = evaluate_gates(
        policy,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        perturbation_drop=0.05,
        champion_oos_sharpe=0.95,
    )
    assert passing_res["promote"] is True
    assert passing_res["failed_gates"] == []
    assert passing_res["checks"]["perturbation_stability"]["passed"] is True


def test_run_cycle_records_git_commit_on_ledger_row(tmp_path):
    # Prepare isolated workspace with symlinks/copies to config/ and data/
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "risk_policy.json").write_text(
        (ROOT / "config" / "risk_policy.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    for fname in ("etf_daily_total_return.csv", "manifest.json"):
        (tmp_path / "data" / fname).write_bytes((ROOT / "data" / fname).read_bytes())

    strat_path = ROOT / "src" / "strategies" / "v000_baseline.py"
    record = run_cycle(
        cycle_id="cycle_000",
        strategy_name="v000_equal_weight_monthly",
        strategy_file=strat_path,
        strategy_fn=v000_weights,
        params={"rebalance_every_days": 21},
        hypothesis="Baseline test",
        neighbor_params_list=[{"rebalance_every_days": 42}],
        is_baseline=True,
        root_dir=tmp_path,
    )
    assert "git_commit" in record
    assert record["git_commit"] == "unknown"  # tmp_path is not a git repo

    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 1
    assert ledger_rows[0]["git_commit"] == "unknown"
