"""Research cycle runner: evaluates candidate vs benchmark & champion, appends ledger, updates champion."""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtest import compute_deflated_sharpe
from engine.data_loader import load_frozen_prices
from engine.policy import RiskPolicy, evaluate_gates, load_policy
from engine.reasoning import format_reasoning_trace_markdown, generate_reasoning_trace
from engine.validation import StrategyFn, evaluate_perturbation_stability, evaluate_walk_forward


def _spy_benchmark_weights(prices: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    bench = str(params["benchmark_ticker"]).upper()
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[bench] = 1.0
    return w


def read_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_champion_metrics(best_dir: Path) -> dict[str, Any] | None:
    p = best_dir / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run_cycle(
    *,
    cycle_id: str,
    strategy_name: str,
    strategy_file: Path,
    strategy_fn: StrategyFn,
    params: dict[str, Any],
    hypothesis: str,
    neighbor_params_list: list[dict[str, Any]] | None = None,
    is_baseline: bool = False,
    root_dir: str | Path = ".",
) -> dict[str, Any]:
    """Execute one bounded research cycle on the development window and log to experiments/ledger.jsonl."""
    t0 = time.perf_counter()
    root = Path(root_dir)
    policy: RiskPolicy = load_policy(root / "config" / "risk_policy.json")

    splits = policy.splits
    dev_prices, manifest = load_frozen_prices(
        root / "data",
        tickers=policy.universe,
        start=splits["development_start"],
        end=splits["development_end"],
        verify_hash=True,
    )

    ledger_path = root / "experiments" / "ledger.jsonl"
    traces_path = root / "experiments" / "reasoning_traces.md"
    best_dir = root / "experiments" / "best_candidate"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    prior_rows = read_ledger(ledger_path)
    trial_count = len(prior_rows) + 1
    champion = load_champion_metrics(best_dir)
    champion_oos_sharpe = float(champion["out_of_sample"]["sharpe"]) if champion else None

    # Benchmark evaluation (SPY buy-and-hold, exempt from 50% single-asset cap).
    bench_eval = evaluate_walk_forward(
        _spy_benchmark_weights,
        dev_prices,
        {"benchmark_ticker": policy.benchmark_ticker},
        policy,
        enforce_single_asset_cap=False,
    )

    # Candidate evaluation
    cand_eval = evaluate_walk_forward(strategy_fn, dev_prices, params, policy)
    oos = cand_eval["out_of_sample"]
    base_oos_sharpe = float(oos["sharpe"])
    perturbation_drop = evaluate_perturbation_stability(
        strategy_fn,
        dev_prices,
        params,
        neighbor_params_list or [],
        policy,
        base_oos_sharpe,
    )

    gate_result = evaluate_gates(
        policy,
        is_metrics=cand_eval["in_sample"],
        oos_metrics=oos,
        perturbation_drop=perturbation_drop,
        champion_oos_sharpe=champion_oos_sharpe,
        is_baseline=is_baseline,
    )

    deflated_sharpe = compute_deflated_sharpe(
        base_oos_sharpe,
        observations=int(oos["observations"]),
        trial_count=trial_count,
        skewness=float(oos.get("skewness", 0.0)),
        excess_kurtosis=float(oos.get("excess_kurtosis", 0.0)),
    )

    decision = "PROMOTE" if gate_result["promote"] else "DISCARD"
    runtime_sec = round(time.perf_counter() - t0, 3)

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle_id": cycle_id,
        "trial_count": trial_count,
        "strategy_name": strategy_name,
        "strategy_file": str(strategy_file.relative_to(root)) if strategy_file.is_relative_to(root) else str(strategy_file),
        "params": params,
        "hypothesis": hypothesis,
        "policy_sha256": policy.sha256,
        "dataset_sha256": manifest["sha256"],
        "runtime_sec": runtime_sec,
        "in_sample": cand_eval["in_sample"],
        "out_of_sample": oos,
        "deflated_sharpe": deflated_sharpe,
        "fold_count": cand_eval["fold_count"],
        "folds": cand_eval["folds"],
        "fold_consistency": cand_eval.get("fold_consistency", {}),
        "attribution": cand_eval.get("attribution", {}),
        "regime_breakdown": cand_eval.get("regime_breakdown", {}),
        "cost_sensitivity": cand_eval.get("cost_sensitivity", {}),
        "benchmark_oos": bench_eval["out_of_sample"],
        "perturbation_sharpe_drop": perturbation_drop,
        "gates": gate_result,
        "decision": decision,
    }
    record["reasoning_trace"] = generate_reasoning_trace(record, prior_champion=champion)

    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    if not traces_path.exists():
        traces_path.write_text("# Experiment Reasoning Traces\n\n", encoding="utf-8")
    with traces_path.open("a", encoding="utf-8") as fh:
        fh.write(format_reasoning_trace_markdown(record))

    if decision == "PROMOTE":
        shutil.copy2(strategy_file, best_dir / "strategy.py")
        (best_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _write_champion_memo(best_dir / "MEMO.md", record, prior_champion=champion)

    return record


def _write_champion_memo(
    memo_path: Path,
    record: dict[str, Any],
    *,
    prior_champion: dict[str, Any] | None,
) -> None:
    oos = record["out_of_sample"]
    ins = record["in_sample"]
    bench = record["benchmark_oos"]
    prior_line = (
        f"Prior champion: `{prior_champion['strategy_name']}` (OOS Sharpe `{prior_champion['out_of_sample']['sharpe']:.3f}`)"
        if prior_champion
        else "Prior champion: none (initial Cycle 0 baseline)"
    )

    lines = [
        f"# Champion Memo — `{record['strategy_name']}` ({record['cycle_id']})",
        "",
        f"- **Promoted at:** `{record['timestamp_utc']}` (trial #{record['trial_count']})",
        f"- **Policy SHA-256:** `{record['policy_sha256'][:16]}...` (verified immutable)",
        f"- **Dataset SHA-256:** `{record['dataset_sha256'][:16]}...` (development window only; holdout locked)",
        f"- **{prior_line}**",
        f"- **Hypothesis:** {record['hypothesis']}",
        f"- **Parameters:** `{json.dumps(record['params'])}`",
        "",
        "## Purged Walk-Forward Out-of-Sample vs. SPY Benchmark (Net of 5 bps/side)",
        "",
        "| Metric | Candidate (OOS) | Candidate (IS) | SPY Buy & Hold (OOS) |",
        "| --- | ---: | ---: | ---: |",
        f"| CAGR | {oos['cagr']*100:.2f}% | {ins['cagr']*100:.2f}% | {bench['cagr']*100:.2f}% |",
        f"| Volatility | {oos['annual_volatility']*100:.2f}% | {ins['annual_volatility']*100:.2f}% | {bench['annual_volatility']*100:.2f}% |",
        f"| Sharpe (net) | {oos['sharpe']:.3f} | {ins['sharpe']:.3f} | {bench['sharpe']:.3f} |",
        f"| Sortino | {oos['sortino']:.3f} | {ins['sortino']:.3f} | {bench['sortino']:.3f} |",
        f"| Max Drawdown | {oos['max_drawdown']*100:.2f}% | {ins['max_drawdown']*100:.2f}% | {bench['max_drawdown']*100:.2f}% |",
        f"| Calmar | {oos['calmar']:.3f} | {ins['calmar']:.3f} | {bench['calmar']:.3f} |",
        f"| Daily 95% CVaR | {oos.get('cvar_95_daily', 0.0)*100:.2f}% | {ins.get('cvar_95_daily', 0.0)*100:.2f}% | {bench.get('cvar_95_daily', 0.0)*100:.2f}% |",
        f"| Annual Turnover | {oos['annual_turnover']*100:.1f}% | {ins['annual_turnover']*100:.1f}% | {bench['annual_turnover']*100:.1f}% |",
        f"| Rebalance Events | {oos['rebalance_events']} | {ins['rebalance_events']} | {bench['rebalance_events']} |",
        "",
        "## Walk-Forward Fold Breakdown",
        "",
        "| Fold | Window | CAGR | Sharpe | Max DD |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for f in record["folds"]:
        lines.append(
            f"| {f['fold']} | `{f['start_date']}` to `{f['end_date']}` | "
            f"{f['cagr']*100:.2f}% | {f['sharpe']:.3f} | {f['max_drawdown']*100:.2f}% |"
        )

    regimes = record.get("regime_breakdown") or {}
    if regimes:
        lines += [
            "",
            "## Stress-Regime Breakdown (Development OOS)",
            "",
            "| Regime | Window | Total Return | Max DD | Sharpe | Worst Day |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for rname, rm in regimes.items():
            lines.append(
                f"| `{rname}` | `{rm['start_date']}` to `{rm['end_date']}` | "
                f"{rm['total_return']*100:.2f}% | {rm['max_drawdown']*100:.2f}% | "
                f"{rm['sharpe']:.3f} | {rm['worst_day']*100:.2f}% |"
            )

    costs = record.get("cost_sensitivity") or {}
    if costs:
        lines += [
            "",
            "## Transaction-Cost Sensitivity Sweep (OOS)",
            "",
            "| Cost Scenario | bps / side | OOS CAGR | OOS Sharpe | OOS Max DD |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for cname, cm in costs.items():
            lines.append(
                f"| `{cname}` | {cm['cost_bps_per_side']:.1f} | {cm['oos_cagr']*100:.2f}% | "
                f"{cm['oos_sharpe']:.3f} | {cm['oos_max_drawdown']*100:.2f}% |"
            )

    attrib = record.get("attribution") or {}
    if attrib and attrib.get("avg_weights"):
        lines += [
            "",
            "## Portfolio Attribution (Development OOS)",
            "",
            (
                f"- **Exposure:** avg gross `{attrib['avg_gross_exposure']*100:.1f}%`, "
                f"cash `{attrib['avg_cash_weight']*100:.1f}%`, "
                f"effective bets `{attrib['avg_effective_bets']:.2f}`, "
                f"SPY beta `{attrib['benchmark_beta']:.2f}` (corr `{attrib['benchmark_correlation']:.2f}`)"
            ),
            "",
            "| Asset | Avg Weight | Annualized Contribution |",
            "| --- | ---: | ---: |",
        ]
        contribs = attrib.get("annualized_contribution") or {}
        for ticker, avg_w in attrib["avg_weights"].items():
            c_val = float(contribs.get(ticker, 0.0))
            lines.append(f"| `{ticker}` | {avg_w*100:.1f}% | {c_val*100:+.2f}% |")

    trace = record.get("reasoning_trace") or {}
    if trace.get("summary_lines"):
        lines += ["", "## Concise Reasoning Trace", ""]
        for item in trace["summary_lines"]:
            lines.append(f"- {item}")

    lines += [
        "",
        "> Offline historical simulation only — not investment advice or live trading instructions.",
        "",
    ]
    memo_path.write_text("\n".join(lines), encoding="utf-8")
