"""Autoresearch-style tab-separated results.tsv exporter and crash record logger."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.runner import read_ledger
from tools.complexity import analyze_strategy_complexity

TSV_HEADER = (
    "commit\tcycle_id\toos_sharpe\toos_max_dd\tannual_turnover\tsloc\tparams\tstatus\tdescription"
)


def get_short_git_commit(root_dir: str | Path = ".") -> str:
    """Return the 7-char short git commit hash, or 'unknown' outside a git repository."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sanitize_tsv_field(text: str, max_len: int = 120) -> str:
    cleaned = " ".join(str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def _decision_to_status(decision: str) -> str:
    upper = str(decision).strip().upper()
    if upper in {"PROMOTE", "KEEP"}:
        return "keep"
    if upper == "CRASH":
        return "crash"
    return "discard"


def format_tsv_row(record: dict[str, Any], *, root_dir: str | Path = ".") -> str:
    """Format a single ledger record into an autoresearch-style TSV row."""
    root = Path(root_dir)
    commit = str(record.get("git_commit") or get_short_git_commit(root))
    cycle_id = _sanitize_tsv_field(str(record.get("cycle_id", "unknown")), max_len=32)
    status = _decision_to_status(str(record.get("decision", "DISCARD")))

    oos = record.get("out_of_sample") or {}
    if status == "crash":
        oos_sharpe = 0.0
        oos_max_dd = 0.0
        ann_to = 0.0
    else:
        oos_sharpe = float(oos.get("sharpe", 0.0))
        oos_max_dd = float(oos.get("max_drawdown", 0.0))
        ann_to = float(oos.get("annual_turnover", 0.0))

    comp = record.get("complexity")
    if not isinstance(comp, dict):
        strat_rel = record.get("strategy_file", "")
        strat_path = (root / strat_rel) if strat_rel else Path()
        comp = analyze_strategy_complexity(strat_path, record.get("params"))

    sloc = int(comp.get("sloc", 0))
    param_count = int(comp.get("param_count", 0))

    strat_name = str(record.get("strategy_name", "")).strip()
    hyp = str(record.get("hypothesis", "")).strip()
    if status == "crash" and record.get("error"):
        desc_raw = f"{strat_name}: CRASH ({record['error']})"
    elif strat_name and hyp:
        desc_raw = f"{strat_name} — {hyp}"
    else:
        desc_raw = strat_name or hyp or "experiment"
    desc = _sanitize_tsv_field(desc_raw, max_len=140)

    return (
        f"{commit}\t{cycle_id}\t{oos_sharpe:.6f}\t{oos_max_dd:.6f}\t"
        f"{ann_to:.4f}\t{sloc}\t{param_count}\t{status}\t{desc}"
    )


def sync_results_tsv(
    ledger_path: str | Path,
    tsv_path: str | Path,
    *,
    root_dir: str | Path = ".",
) -> int:
    """Regenerate experiments/results.tsv from experiments/ledger.jsonl. Returns row count."""
    ledger = Path(ledger_path)
    tsv = Path(tsv_path)
    rows = read_ledger(ledger)
    lines = [TSV_HEADER]
    for rec in rows:
        lines.append(format_tsv_row(rec, root_dir=root_dir))
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def append_crash_record(
    *,
    cycle_id: str,
    strategy_name: str,
    strategy_file: Path,
    params: dict[str, Any],
    hypothesis: str,
    error_message: str,
    runtime_sec: float,
    root_dir: str | Path = ".",
) -> dict[str, Any]:
    """Append a CRASH record to experiments/ledger.jsonl and refresh experiments/results.tsv."""
    root = Path(root_dir)
    ledger_path = root / "experiments" / "ledger.jsonl"
    tsv_path = root / "experiments" / "results.tsv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    prior_rows = read_ledger(ledger_path)
    trial_count = len(prior_rows) + 1
    complexity = analyze_strategy_complexity(strategy_file, params)

    rel_file = (
        str(strategy_file.relative_to(root))
        if strategy_file.is_relative_to(root)
        else str(strategy_file)
    )
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle_id": cycle_id,
        "trial_count": trial_count,
        "git_commit": get_short_git_commit(root),
        "strategy_name": strategy_name,
        "strategy_file": rel_file,
        "params": params,
        "hypothesis": hypothesis,
        "runtime_sec": round(runtime_sec, 3),
        "complexity": complexity,
        "out_of_sample": {},
        "in_sample": {},
        "gates": {"promote": False, "risk_passed": False, "failed_gates": ["crash"]},
        "decision": "CRASH",
        "error": _sanitize_tsv_field(error_message, max_len=240),
    }

    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    sync_results_tsv(ledger_path, tsv_path, root_dir=root)
    return record
