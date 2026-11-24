"""Offline unit tests for autoresearch-inspired tooling (complexity, results.tsv, analyzer, CLI)."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.analyze_experiments import (
    compute_experiment_summary,
    format_terminal_report,
    render_progress_svg,
)
from tools.complexity import analyze_strategy_complexity
from tools.experiment_cli import check_immutable_harness_clean, run_experiment_cli
from tools.results_tsv import TSV_HEADER, append_crash_record, format_tsv_row, sync_results_tsv

ROOT = Path(__file__).resolve().parents[1]


def test_complexity_auditor_counts_sloc_params_and_branches():
    v000 = analyze_strategy_complexity(ROOT / "src" / "strategies" / "v000_baseline.py")
    v006 = analyze_strategy_complexity(ROOT / "src" / "strategies" / "v006_vol_targeted_inv_vol.py")
    assert v000["sloc"] > 0
    assert v000["param_count"] >= 1
    assert v006["sloc"] > v000["sloc"]
    assert v006["param_count"] > v000["param_count"]
    assert v006["branch_count"] > 0


def test_sync_results_tsv_and_crash_logging(tmp_path):
    ledger = tmp_path / "experiments" / "ledger.jsonl"
    tsv = tmp_path / "experiments" / "results.tsv"
    ledger.parent.mkdir(parents=True)

    strat_file = tmp_path / "strat_ok.py"
    strat_file.write_text("DEFAULT_PARAMS = {'a': 1, 'b': 2}\ndef generate_weights(p, d): return p\n", encoding="utf-8")

    records = [
        {
            "cycle_id": "cycle_000",
            "git_commit": "abc1234",
            "strategy_name": "v000_base",
            "strategy_file": "strat_ok.py",
            "params": {"a": 1},
            "hypothesis": "Baseline\twith\ttabs\nand newlines",
            "out_of_sample": {"sharpe": 0.95, "max_drawdown": -0.20, "annual_turnover": 0.25},
            "decision": "PROMOTE",
        },
        {
            "cycle_id": "cycle_001",
            "git_commit": "def5678",
            "strategy_name": "v001_bad",
            "strategy_file": "strat_ok.py",
            "params": {"a": 1, "b": 2},
            "hypothesis": "Failed attempt",
            "out_of_sample": {"sharpe": 0.80, "max_drawdown": -0.30, "annual_turnover": 1.2},
            "decision": "DISCARD",
        },
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    n_rows = sync_results_tsv(ledger, tsv, root_dir=tmp_path)
    assert n_rows == 2

    crash_rec = append_crash_record(
        cycle_id="cycle_002",
        strategy_name="v002_boom",
        strategy_file=strat_file,
        params={"a": 1},
        hypothesis="Blows up",
        error_message="ZeroDivisionError: division by zero",
        runtime_sec=0.12,
        root_dir=tmp_path,
    )
    assert crash_rec["decision"] == "CRASH"

    lines = tsv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == TSV_HEADER
    assert len(lines) == 4
    header = TSV_HEADER.split("\t")
    assert header.index("strategy_note") == header.index("strategy") + 1
    expected_cols = len(header)
    for row in lines[1:]:
        cols = row.split("\t")
        assert len(cols) == expected_cols
    status_idx = header.index("status")
    sharpe_idx = header.index("oos_sharpe")
    assert lines[1].split("\t")[status_idx] == "keep"
    assert lines[2].split("\t")[status_idx] == "discard"
    assert lines[3].split("\t")[status_idx] == "crash"
    assert lines[3].split("\t")[sharpe_idx] == "0.000000"


def test_analyzer_summary_and_valid_svg_chart(tmp_path):
    rows = [
        {
            "cycle_id": "cycle_000",
            "strategy_name": "v000_base",
            "out_of_sample": {"sharpe": 0.90, "max_drawdown": -0.22, "annual_turnover": 0.2},
            "benchmark_oos": {"sharpe": 0.80},
            "complexity": {"sloc": 30, "param_count": 1},
            "decision": "PROMOTE",
            "gates": {"failed_gates": []},
        },
        {
            "cycle_id": "cycle_001",
            "strategy_name": "v001_better",
            "out_of_sample": {"sharpe": 1.05, "max_drawdown": -0.15, "annual_turnover": 0.4},
            "benchmark_oos": {"sharpe": 0.80},
            "complexity": {"sloc": 45, "param_count": 4},
            "decision": "PROMOTE",
            "gates": {"failed_gates": []},
        },
        {
            "cycle_id": "cycle_002",
            "strategy_name": "v002_discard",
            "out_of_sample": {"sharpe": 0.98, "max_drawdown": -0.18, "annual_turnover": 0.8},
            "benchmark_oos": {"sharpe": 0.80},
            "decision": "DISCARD",
            "gates": {"failed_gates": ["beats_champion_margin"]},
        },
        {
            "cycle_id": "cycle_003",
            "strategy_name": "v003_crash",
            "out_of_sample": {},
            "decision": "CRASH",
            "gates": {"failed_gates": ["crash"]},
        },
    ]

    summary = compute_experiment_summary(rows, root_dir=tmp_path)
    assert summary["total_experiments"] == 4
    assert summary["n_keep"] == 2
    assert summary["n_discard"] == 1
    assert summary["n_crash"] == 1
    assert summary["consecutive_discards"] == 2
    assert summary["total_improvement"] == 0.15
    assert summary["gate_failures"]["beats_champion_margin"] == 1
    assert len(summary["top_hits"]) == 1
    assert summary["top_hits"][0]["delta_sharpe"] == 0.15

    report = format_terminal_report(summary, stop_after_consecutive_discards=3)
    assert "Top Hits" in report
    assert "Discard Gate Breakdown" in report
    assert "v001_better" in report

    svg_path = render_progress_svg(rows, tmp_path / "progress.svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    tree = ET.fromstring(svg_text)
    assert tree.tag.endswith("svg")
    assert "SPY OOS Sharpe" in svg_text
    assert "v001_better" in svg_text


def test_experiment_cli_dry_run_and_crash_handling(tmp_path, capsys):
    # Dry-run on real baseline strategy should succeed without mutating repo state
    rc = run_experiment_cli(
        [
            "--root",
            str(ROOT),
            "--strategy",
            "src/strategies/v000_baseline.py",
            "--dry-run",
            "--baseline",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "---" in out
    assert "oos_sharpe:" in out
    assert "sloc:" in out
    assert "decision:" in out

    # Crashing strategy in isolated temp dir should return 1 and log CRASH to ledger & results.tsv
    bad_strat = tmp_path / "bad_strategy.py"
    bad_strat.write_text(
        "STRATEGY_NAME = 'v999_broken'\n"
        "DEFAULT_PARAMS = {'x': 1}\n"
        "def generate_weights(prices, params):\n"
        "    raise RuntimeError('intentional failure')\n",
        encoding="utf-8",
    )
    rc_crash = run_experiment_cli(
        [
            "--root",
            str(tmp_path),
            "--strategy",
            str(bad_strat),
        ]
    )
    assert rc_crash == 1
    crash_out = capsys.readouterr().out
    assert "status:              crash" in crash_out
    assert "decision:            CRASH" in crash_out
    assert (tmp_path / "experiments" / "ledger.jsonl").exists()
    assert (tmp_path / "experiments" / "results.tsv").exists()


def test_format_tsv_row_uses_unknown_when_git_commit_missing(tmp_path):
    strat_file = tmp_path / "strat.py"
    strat_file.write_text("DEFAULT_PARAMS = {'x': 1}\ndef generate_weights(p, d): return p\n", encoding="utf-8")

    # Even when root_dir is the real git repo, a row without git_commit must not stamp current HEAD.
    row_missing = {
        "cycle_id": "cycle_000",
        "strategy_name": "v000_base",
        "strategy_file": str(strat_file),
        "params": {"x": 1},
        "hypothesis": "Missing commit field",
        "out_of_sample": {"sharpe": 0.90, "max_drawdown": -0.15, "annual_turnover": 0.2},
        "decision": "PROMOTE",
    }
    row_empty = {**row_missing, "git_commit": ""}
    assert format_tsv_row(row_missing, root_dir=ROOT).split("\t")[0] == "unknown"
    assert format_tsv_row(row_empty, root_dir=ROOT).split("\t")[0] == "unknown"


def test_check_immutable_harness_clean_watches_data_and_fails_closed(tmp_path, monkeypatch):
    # Initialize an isolated git repo with tracked files under config/, src/engine/, and data/.
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    for rel, content in [
        ("config/risk_policy.json", "{}\n"),
        ("src/engine/backtest.py", "# clean\n"),
        ("data/etf_daily_total_return.csv", "date,SPY\n2020-01-02,100\n"),
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    assert check_immutable_harness_clean(tmp_path) == []

    # Modifying data/ must be detected.
    (tmp_path / "data" / "etf_daily_total_return.csv").write_text("tampered\n", encoding="utf-8")
    dirty = check_immutable_harness_clean(tmp_path)
    assert any("data/etf_daily_total_return.csv" in item for item in dirty)

    # Git failure must fail closed (raise RuntimeError, not return []).
    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=["git", "status"], stderr="not a git repo")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(RuntimeError, match="Immutable harness git check failed"):
        check_immutable_harness_clean(tmp_path)


def test_cli_enforces_stop_after_consecutive_discards_with_override(tmp_path, monkeypatch, capsys):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    policy_raw = json.loads((ROOT / "config" / "risk_policy.json").read_text(encoding="utf-8"))
    policy_raw["promotion"]["stop_after_consecutive_discards"] = 3
    (cfg_dir / "risk_policy.json").write_text(json.dumps(policy_raw), encoding="utf-8")

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True)
    ledger_path = exp_dir / "ledger.jsonl"
    discard_rows = [
        {"cycle_id": f"cycle_{i:03d}", "decision": "DISCARD"}
        for i in range(3)
    ]
    ledger_path.write_text("\n".join(json.dumps(r) for r in discard_rows) + "\n", encoding="utf-8")

    strat_file = tmp_path / "strat.py"
    strat_file.write_text(
        "STRATEGY_NAME = 'v999_test'\n"
        "DEFAULT_PARAMS = {'x': 1}\n"
        "NEIGHBOR_PARAMS = [{'x': 2}]\n"
        "def generate_weights(prices, params):\n"
        "    return prices * 0.0\n",
        encoding="utf-8",
    )

    # Without override, CLI must abort non-zero and NOT append to ledger.jsonl.
    rc_blocked = run_experiment_cli(["--root", str(tmp_path), "--strategy", str(strat_file)])
    err_out = capsys.readouterr().err
    assert rc_blocked == 2
    assert "stop_after_consecutive_discards=3" in err_out
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 3

    # With --allow-after-consecutive-discards, the guard passes and proceeds to run_cycle.
    called = {}

    def _fake_run_cycle(**kwargs):
        called["cycle_id"] = kwargs["cycle_id"]
        return {
            "cycle_id": kwargs["cycle_id"],
            "decision": "DISCARD",
            "runtime_sec": 0.01,
            "in_sample": {"sharpe": 0.5},
            "out_of_sample": {"sharpe": 0.4, "cagr": 0.03, "max_drawdown": -0.1, "annual_turnover": 0.2},
            "gates": {"failed_gates": ["beats_champion_margin"]},
            "deflated_sharpe": 0.5,
            "attribution": {},
            "reasoning_trace": {"summary_lines": ["stub"]},
        }

    monkeypatch.setattr("tools.experiment_cli.run_cycle", _fake_run_cycle)
    monkeypatch.setattr("tools.experiment_cli.sync_results_tsv", lambda *a, **kw: 4)

    rc_allowed = run_experiment_cli(
        [
            "--root",
            str(tmp_path),
            "--strategy",
            str(strat_file),
            "--allow-after-consecutive-discards",
        ]
    )
    capsys.readouterr()
    assert rc_allowed == 0
    assert called["cycle_id"] == "cycle_003"
