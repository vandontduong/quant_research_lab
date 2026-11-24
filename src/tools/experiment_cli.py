"""Single-command experiment CLI inspired by karpathy/autoresearch.

Runs a candidate strategy through quant_research_lab's purged walk-forward harness with:
- Optional immutable-harness git check (config/risk_policy.json, src/engine/, data/ untouched)
- Wall-clock timeout enforcement
- Automatic AST/parameter complexity audit (simplicity criterion)
- Automatic CRASH logging & results.tsv syncing (when not --dry-run)
- Grep-friendly `---` summary block matching autoresearch's stdout convention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from engine.backtest import compute_deflated_sharpe
from engine.data_loader import load_frozen_prices
from engine.policy import RiskPolicy, evaluate_gates, load_policy
from engine.reasoning import generate_reasoning_trace
from engine.runner import (
    count_trailing_consecutive_discards,
    load_champion_metrics,
    read_ledger,
    run_cycle,
)
from engine.validation import evaluate_perturbation_stability, evaluate_walk_forward
from tools.complexity import analyze_strategy_complexity
from tools.results_tsv import append_crash_record, get_short_git_commit, sync_results_tsv


class ExperimentTimeoutError(TimeoutError):
    """Raised when an experiment exceeds its wall-clock time budget."""


@contextmanager
def wall_clock_timeout(seconds: int):
    """Enforce a wall-clock timeout using SIGALRM on Unix platforms."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise ExperimentTimeoutError(f"Experiment exceeded wall-clock timeout of {seconds}s.")

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def check_immutable_harness_clean(root_dir: Path) -> list[str]:
    """Return modified/untracked files under immutable paths (config/risk_policy.json, src/engine/, data/).

    Fails closed by raising RuntimeError if git status cannot be verified.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", "config/risk_policy.json", "src/engine", "data"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        stderr = getattr(exc, "stderr", None)
        detail = f" ({str(stderr).strip()})" if stderr else ""
        raise RuntimeError(f"Immutable harness git check failed in {root_dir}: {exc}{detail}") from exc
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def check_consecutive_discards_stop(
    ledger_path: Path,
    policy: RiskPolicy,
    *,
    allow_override: bool = False,
) -> None:
    """Abort with RuntimeError if trailing non-keep runs reached policy stop_after_consecutive_discards."""
    if allow_override:
        return
    limit = int(policy.promotion.get("stop_after_consecutive_discards", 0))
    if limit <= 0:
        return
    trailing = count_trailing_consecutive_discards(read_ledger(ledger_path))
    if trailing >= limit:
        raise RuntimeError(
            f"Trailing consecutive non-keep runs ({trailing}) reached policy limit "
            f"stop_after_consecutive_discards={limit}. Pass --allow-after-consecutive-discards to override."
        )


def load_strategy_module(strategy_file: Path) -> Any:
    """Dynamically load a strategy Python file and verify it exposes generate_weights."""
    resolved = strategy_file.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Strategy file not found: {resolved}")
    module_name = f"qrl_strategy_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate_weights"):
        raise AttributeError(f"Strategy module {resolved.name} must define `generate_weights(prices, params)`.")
    return mod


def _next_cycle_id(ledger_path: Path) -> str:
    rows = read_ledger(ledger_path)
    return f"cycle_{len(rows):03d}"


def print_summary_block(summary: dict[str, Any]) -> None:
    """Print an autoresearch-style grep-friendly key: value summary block."""
    print("---")
    for key, val in summary.items():
        print(f"{key + ':':<20} {val}")


def run_experiment_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single quant_research_lab experiment with autoresearch-style output & logging."
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        required=True,
        help="Path to candidate strategy module (e.g. src/strategies/v006_vol_targeted_inv_vol.py).",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory.")
    parser.add_argument("--cycle-id", type=str, default=None, help="Explicit cycle ID (default: auto-increment).")
    parser.add_argument("--hypothesis", type=str, default=None, help="Override strategy HYPOTHESIS string.")
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="JSON object overriding DEFAULT_PARAMS.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Wall-clock timeout in seconds (0 disables timeout; default: 120).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Mark this run as Cycle 0 baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate candidate without appending to ledger.jsonl, updating best_candidate, or syncing results.tsv.",
    )
    parser.add_argument(
        "--check-git-clean-harness",
        action="store_true",
        help="Fail fast if config/risk_policy.json, src/engine/, or data/ has uncommitted git modifications.",
    )
    parser.add_argument(
        "--allow-after-consecutive-discards",
        action="store_true",
        help="Allow running a new non-dry-run experiment after reaching stop_after_consecutive_discards.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    strategy_path = args.strategy if args.strategy.is_absolute() else (root / args.strategy)
    strategy_path = strategy_path.resolve()

    if args.check_git_clean_harness:
        try:
            dirty = check_immutable_harness_clean(root)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if dirty:
            print(f"ERROR: Immutable harness files modified: {dirty}", file=sys.stderr)
            return 2

    ledger_path = root / "experiments" / "ledger.jsonl"
    tsv_path = root / "experiments" / "results.tsv"

    if not args.dry_run:
        policy_path = root / "config" / "risk_policy.json"
        if policy_path.exists():
            try:
                check_consecutive_discards_stop(
                    ledger_path,
                    load_policy(policy_path),
                    allow_override=args.allow_after_consecutive_discards,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

    cycle_id = args.cycle_id or _next_cycle_id(ledger_path)
    git_commit = get_short_git_commit(root)

    t0 = time.perf_counter()
    strategy_name = strategy_path.stem
    params: dict[str, Any] = {}
    hypothesis = args.hypothesis or ""

    try:
        mod = load_strategy_module(strategy_path)
        strategy_name = str(getattr(mod, "STRATEGY_NAME", strategy_path.stem))
        base_params = dict(getattr(mod, "DEFAULT_PARAMS", {}))
        if args.params_json:
            overrides = json.loads(args.params_json)
            if not isinstance(overrides, dict):
                raise ValueError("--params-json must decode to a JSON object.")
            base_params.update(overrides)
        params = base_params
        hypothesis = args.hypothesis or str(getattr(mod, "HYPOTHESIS", "No hypothesis provided."))
        neighbor_params = getattr(mod, "NEIGHBOR_PARAMS", None)
        complexity = analyze_strategy_complexity(strategy_path, params)

        with wall_clock_timeout(args.timeout_sec):
            if args.dry_run:
                policy = load_policy(root / "config" / "risk_policy.json")
                dev_prices, _ = load_frozen_prices(
                    root / "data",
                    tickers=policy.universe,
                    start=policy.splits["development_start"],
                    end=policy.splits["development_end"],
                    verify_hash=True,
                )
                cand_eval = evaluate_walk_forward(mod.generate_weights, dev_prices, params, policy)
                oos = cand_eval["out_of_sample"]
                ins = cand_eval["in_sample"]
                oos_sharpe = float(oos["sharpe"])
                pdrop = evaluate_perturbation_stability(
                    mod.generate_weights,
                    dev_prices,
                    params,
                    neighbor_params or [],
                    policy,
                    oos_sharpe,
                )
                champ = load_champion_metrics(root / "experiments" / "best_candidate")
                champ_sharpe = float(champ["out_of_sample"]["sharpe"]) if champ else None
                gates = evaluate_gates(
                    policy,
                    is_metrics=ins,
                    oos_metrics=oos,
                    perturbation_drop=pdrop,
                    champion_oos_sharpe=champ_sharpe,
                    is_baseline=args.baseline,
                )
                trial_count = len(read_ledger(ledger_path)) + 1
                dsr = compute_deflated_sharpe(
                    oos_sharpe,
                    observations=int(oos["observations"]),
                    trial_count=trial_count,
                    skewness=float(oos.get("skewness", 0.0)),
                    excess_kurtosis=float(oos.get("excess_kurtosis", 0.0)),
                )
                decision = "PROMOTE" if gates["promote"] else "DISCARD"
                runtime_sec = round(time.perf_counter() - t0, 3)
                attrib = cand_eval.get("attribution") or {}
                synthetic_record = {
                    "cycle_id": cycle_id,
                    "trial_count": trial_count,
                    "strategy_name": strategy_name,
                    "in_sample": ins,
                    "out_of_sample": oos,
                    "deflated_sharpe": dsr,
                    "folds": cand_eval["folds"],
                    "fold_consistency": cand_eval.get("fold_consistency", {}),
                    "attribution": attrib,
                    "regime_breakdown": cand_eval.get("regime_breakdown", {}),
                    "cost_sensitivity": cand_eval.get("cost_sensitivity", {}),
                    "perturbation_sharpe_drop": pdrop,
                    "gates": gates,
                    "decision": decision,
                }
                trace = generate_reasoning_trace(
                    synthetic_record,
                    prior_champion=champ,
                    complexity=complexity,
                )
            else:
                record = run_cycle(
                    cycle_id=cycle_id,
                    strategy_name=strategy_name,
                    strategy_file=strategy_path,
                    strategy_fn=mod.generate_weights,
                    params=params,
                    hypothesis=hypothesis,
                    neighbor_params_list=neighbor_params,
                    is_baseline=args.baseline,
                    root_dir=root,
                )
                sync_results_tsv(ledger_path, tsv_path, root_dir=root)
                decision = record["decision"]
                runtime_sec = record["runtime_sec"]
                oos = record["out_of_sample"]
                ins = record["in_sample"]
                gates = record["gates"]
                dsr = record.get("deflated_sharpe")
                attrib = record.get("attribution") or {}
                trace = record.get("reasoning_trace") or generate_reasoning_trace(
                    record, complexity=complexity
                )

        failed_str = ",".join(gates.get("failed_gates", [])) or "none"
        status = "keep" if decision == "PROMOTE" else "discard"
        summary_dict: dict[str, Any] = {
            "cycle_id": cycle_id,
            "strategy_name": strategy_name,
            "git_commit": git_commit,
            "dry_run": str(args.dry_run).lower(),
            "oos_sharpe": f"{float(oos['sharpe']):.6f}",
            "is_sharpe": f"{float(ins['sharpe']):.6f}",
            "dsr_confidence": f"{float(dsr):.6f}" if dsr is not None else "n/a",
            "oos_cagr": f"{float(oos['cagr']):.6f}",
            "oos_max_drawdown": f"{float(oos['max_drawdown']):.6f}",
            "annual_turnover": f"{float(oos['annual_turnover']):.6f}",
            "effective_bets": f"{float(attrib['avg_effective_bets']):.3f}" if "avg_effective_bets" in attrib else "n/a",
            "avg_cash_weight": f"{float(attrib['avg_cash_weight']):.4f}" if "avg_cash_weight" in attrib else "n/a",
            "param_count": complexity["param_count"],
            "sloc": complexity["sloc"],
            "branch_count": complexity["branch_count"],
            "runtime_sec": f"{runtime_sec:.3f}",
            "status": status,
            "decision": decision,
            "failed_gates": failed_str,
        }
        print_summary_block(summary_dict)
        if trace.get("summary_lines"):
            print("reasoning_trace:")
            for line in trace["summary_lines"]:
                print(f"  - {line}")
        return 0

    except Exception as exc:  # noqa: BLE001
        runtime_sec = round(time.perf_counter() - t0, 3)
        err_msg = f"{type(exc).__name__}: {exc}"
        complexity = analyze_strategy_complexity(strategy_path, params or None)
        if not args.dry_run:
            append_crash_record(
                cycle_id=cycle_id,
                strategy_name=strategy_name,
                strategy_file=strategy_path,
                params=params,
                hypothesis=hypothesis or "Crash during execution",
                error_message=err_msg,
                runtime_sec=runtime_sec,
                root_dir=root,
            )
        print_summary_block(
            {
                "cycle_id": cycle_id,
                "strategy_name": strategy_name,
                "git_commit": git_commit,
                "dry_run": str(args.dry_run).lower(),
                "oos_sharpe": "0.000000",
                "param_count": complexity["param_count"],
                "sloc": complexity["sloc"],
                "runtime_sec": f"{runtime_sec:.3f}",
                "status": "crash",
                "decision": "CRASH",
                "error": err_msg,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run_experiment_cli())
