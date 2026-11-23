"""Offline unit tests for autoresearch-inspired tooling (complexity, results.tsv, analyzer, CLI)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.analyze_experiments import (
    compute_experiment_summary,
    format_terminal_report,
    render_progress_svg,
)
from tools.complexity import analyze_strategy_complexity
from tools.experiment_cli import run_experiment_cli
from tools.results_tsv import TSV_HEADER, append_crash_record, sync_results_tsv

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
