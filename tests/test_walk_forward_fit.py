"""Unit tests for per-fold purged walk-forward fitting, CLI fit hooks, and holdout audit guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from engine.policy import load_policy
from engine.runner import run_holdout_audit
from engine.validation import evaluate_perturbation_stability, evaluate_walk_forward
from strategies import v000_baseline
from tools.experiment_cli import run_experiment_cli

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_dev_prices(periods: int = 2600, start: str = "2005-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(101)
    dates = pd.bdate_range(start=start, periods=periods)
    cols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"]
    data = {}
    for idx, col in enumerate(cols):
        rets = rng.normal(0.00035 + idx * 0.00003, 0.009, size=periods)
        data[col] = 100.0 * np.cumprod(1.0 + rets)
    df = pd.DataFrame(data, index=dates)
    assert df.index[-1] < pd.Timestamp("2024-01-02")
    return df


def test_per_fold_purged_walk_forward_respects_purge_and_drives_weights():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_dev_prices(periods=2600)
    purge_days = int(policy.splits["walk_forward_purge_days"])

    seen_train_ends: list[pd.Timestamp] = []
    seen_train_lens: list[int] = []

    def stateful_fit(train_prices: pd.DataFrame, base_params: dict[str, Any]) -> dict[str, Any]:
        # Record exact train boundary passed into fit_fn
        seen_train_ends.append(train_prices.index[-1])
        seen_train_lens.append(len(train_prices))
        # Fit a train-only statistic (trailing mean SPY return scaled into a bounded tilt)
        spy_mean = float(train_prices["SPY"].pct_change().dropna().mean())
        spy_w = float(np.clip(base_params["base_spy_weight"] + spy_mean * 100.0, 0.15, 0.45))
        return {
            **base_params,
            "fitted_spy_weight": round(spy_w, 6),
            "train_last_date": train_prices.index[-1].strftime("%Y-%m-%d"),
            "train_last_spy_close": round(float(train_prices["SPY"].iloc[-1]), 6),
        }

    def fitted_strategy(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        spy_w = float(params["fitted_spy_weight"])
        other_w = (1.0 - spy_w) / (len(df.columns) - 1)
        for col in df.columns:
            w[col] = spy_w if col == "SPY" else other_w
        return w

    res = evaluate_walk_forward(
        fitted_strategy,
        prices,
        {"base_spy_weight": 0.25},
        policy,
        fit_fn=stateful_fit,
    )

    assert res["evaluation_mode"] == "per_fold_purged_wf"
    assert res["fold_count"] >= 2
    assert len(seen_train_ends) == res["fold_count"]
    assert "fitted_params_by_fold" in res
    assert len(res["fitted_params_by_fold"]) == res["fold_count"]

    # (a) Verify fit_fn only ever saw data strictly up to train_end with full 21-day purge gap
    for fold_meta, seen_end, seen_len in zip(
        res["fitted_params_by_fold"], seen_train_ends, seen_train_lens, strict=True
    ):
        train_end_idx = int(fold_meta["train_end"])
        test_start_idx = int(fold_meta["test_start"])
        assert test_start_idx - train_end_idx == purge_days
        assert seen_len == train_end_idx
        assert seen_end == prices.index[train_end_idx - 1]
        assert seen_end < prices.index[test_start_idx]
        # Confirm no purge bar or test bar entered fit_fn
        assert prices.index.get_loc(seen_end) <= test_start_idx - purge_days - 1

    # (b) Per-fold fitted params change across folds and drive fold-specific evaluation
    train_closes = [f["fitted_params"]["train_last_spy_close"] for f in res["fitted_params_by_fold"]]
    assert len(set(train_closes)) == res["fold_count"]
    assert res["folds"][0]["fitted_params"]["train_last_date"] == seen_train_ends[0].strftime("%Y-%m-%d")
    assert "attribution" in res
    assert "cost_sensitivity" in res
    assert "regime_breakdown" in res

    # Perturbation stability threads fit_fn through evaluate_walk_forward cleanly
    pdrop = evaluate_perturbation_stability(
        fitted_strategy,
        prices,
        {"base_spy_weight": 0.25},
        [{"base_spy_weight": 0.20}, {"base_spy_weight": 0.30}],
        policy,
        float(res["out_of_sample"]["sharpe"]),
        fit_fn=stateful_fit,
    )
    assert pdrop is not None and np.isfinite(pdrop)


def test_fixed_param_strategy_reports_fixed_param_causal_oos():
    policy = load_policy(ROOT / "config" / "risk_policy.json")
    prices = _synthetic_dev_prices(periods=2600)
    res = evaluate_walk_forward(
        v000_baseline.generate_weights,
        prices,
        v000_baseline.DEFAULT_PARAMS,
        policy,
        include_diagnostics=False,
    )
    assert res["evaluation_mode"] == "fixed_param_causal_oos"
    assert "fitted_params_by_fold" not in res


def test_cli_detects_fit_strategy_hook_in_dry_run_and_run_cycle(tmp_path, capsys):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "risk_policy.json").write_text(
        (ROOT / "config" / "risk_policy.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    for fname in ("etf_daily_total_return.csv", "manifest.json"):
        (tmp_path / "data" / fname).write_bytes((ROOT / "data" / fname).read_bytes())

    strat_path = tmp_path / "v998_fit_test.py"
    strat_path.write_text(
        "STRATEGY_NAME = 'v998_fit_test'\n"
        "DEFAULT_PARAMS = {'base_w': 0.2}\n"
        "NEIGHBOR_PARAMS = [{'base_w': 0.18}]\n"
        "HYPOTHESIS = 'Test per-fold fit_strategy CLI detection.'\n"
        "def fit_strategy(train_prices, params):\n"
        "    return {**params, 'train_end': train_prices.index[-1].strftime('%Y-%m-%d')}\n"
        "def generate_weights(prices, params):\n"
        "    w = prices * 0.0\n"
        "    w['SPY'] = float(params['base_w'])\n"
        "    w['TLT'] = 0.30\n"
        "    w['SHY'] = 0.30\n"
        "    return w\n",
        encoding="utf-8",
    )

    rc_dry = run_experiment_cli(
        ["--root", str(tmp_path), "--strategy", str(strat_path), "--dry-run", "--baseline"]
    )
    assert rc_dry == 0
    out_dry = capsys.readouterr().out
    assert "evaluation_mode:     per_fold_purged_wf" in out_dry
    assert not (tmp_path / "experiments" / "ledger.jsonl").exists()

    rc_run = run_experiment_cli(
        ["--root", str(tmp_path), "--strategy", str(strat_path), "--baseline"]
    )
    assert rc_run == 0
    capsys.readouterr()
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_rows) == 1
    assert ledger_rows[0]["evaluation_mode"] == "per_fold_purged_wf"
    assert len(ledger_rows[0]["fitted_params_by_fold"]) == ledger_rows[0]["fold_count"]


def test_holdout_audit_refuses_without_confirm_flag(tmp_path):
    out_file = tmp_path / "holdout_audit.json"
    with pytest.raises(PermissionError, match="Holdout window is locked"):
        run_holdout_audit(
            strategy_fn=v000_baseline.generate_weights,
            params=v000_baseline.DEFAULT_PARAMS,
            strategy_name="v000_equal_weight_monthly",
            root_dir=ROOT,
            output_path=out_file,
            confirm_holdout_audit=False,
        )
    assert not out_file.exists()

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_holdout_audit.py"),
            "--strategy",
            "src/strategies/v000_baseline.py",
            "--output",
            str(out_file),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Holdout window is locked" in proc.stderr
    assert not out_file.exists()
