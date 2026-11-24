"""Research cycle runner: evaluates candidate vs benchmark & champion, appends ledger, updates champion."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtest import compute_deflated_sharpe, run_backtest, sharpe_series_moments
from engine.data_loader import load_frozen_prices
from engine.policy import RiskPolicy, evaluate_gates, load_policy
from engine.reasoning import format_reasoning_trace_markdown, generate_reasoning_trace
from engine.validation import FitFn, StrategyFn, evaluate_perturbation_stability, evaluate_walk_forward


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


def count_trailing_consecutive_discards(rows: list[dict[str, Any]]) -> int:
    """Count trailing consecutive non-keep (DISCARD/CRASH) decisions in a ledger."""
    count = 0
    for row in reversed(rows):
        decision = str(row.get("decision", "DISCARD")).strip().upper()
        if decision in {"PROMOTE", "KEEP"}:
            break
        count += 1
    return count


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
    fit_fn: FitFn | None = None,
    is_baseline: bool = False,
    root_dir: str | Path = ".",
) -> dict[str, Any]:
    """Execute one bounded research cycle on the development window and log to experiments/ledger.jsonl.

    Note: `promotion.stop_after_consecutive_discards` is enforced at the CLI gate
    (`tools.experiment_cli.run_experiment_cli`) so programmatic test calls to `run_cycle` remain usable.
    """
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
    cand_eval = evaluate_walk_forward(
        strategy_fn,
        dev_prices,
        params,
        policy,
        fit_fn=fit_fn,
    )
    oos = cand_eval["out_of_sample"]
    base_oos_sharpe = float(oos["sharpe"])
    perturbation_drop = evaluate_perturbation_stability(
        strategy_fn,
        dev_prices,
        params,
        neighbor_params_list or [],
        policy,
        base_oos_sharpe,
        fit_fn=fit_fn,
    )

    dsr_skew, dsr_ex_kurt = sharpe_series_moments(oos)
    deflated_sharpe = compute_deflated_sharpe(
        base_oos_sharpe,
        observations=int(oos["observations"]),
        trial_count=trial_count,
        skewness=dsr_skew,
        excess_kurtosis=dsr_ex_kurt,
    )

    gate_result = evaluate_gates(
        policy,
        is_metrics=cand_eval["in_sample"],
        oos_metrics=oos,
        perturbation_drop=perturbation_drop,
        deflated_sharpe=deflated_sharpe,
        champion_oos_sharpe=champion_oos_sharpe,
        is_baseline=is_baseline,
    )

    decision = "PROMOTE" if gate_result["promote"] else "DISCARD"
    runtime_sec = round(time.perf_counter() - t0, 3)

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cycle_id": cycle_id,
        "trial_count": trial_count,
        "git_commit": get_short_git_commit(root),
        "strategy_name": strategy_name,
        "strategy_file": str(strategy_file.relative_to(root)) if strategy_file.is_relative_to(root) else str(strategy_file),
        "params": params,
        "hypothesis": hypothesis,
        "policy_sha256": policy.sha256,
        "dataset_sha256": manifest["sha256"],
        "evaluation_mode": cand_eval["evaluation_mode"],
        "sharpe_excess_of_cash_proxy": policy.sharpe_excess_of_cash_proxy,
        "cash_proxy_ticker": policy.cash_proxy_ticker,
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
    if "fitted_params_by_fold" in cand_eval:
        record["fitted_params_by_fold"] = cand_eval["fitted_params_by_fold"]
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
    eval_mode = str(record.get("evaluation_mode", "fixed_param_causal_oos"))
    cash_ticker = record.get("cash_proxy_ticker") or "SHY"
    excess_enabled = bool(record.get("sharpe_excess_of_cash_proxy", False))
    dsr_val = record.get("deflated_sharpe")
    pdrop_val = record.get("perturbation_sharpe_drop")
    prior_line = (
        f"Prior champion: `{prior_champion['strategy_name']}` (OOS Sharpe `{prior_champion['out_of_sample']['sharpe']:.3f}`)"
        if prior_champion
        else "Prior champion: none (initial Cycle 0 baseline)"
    )
    scoring_line = (
        f"Excess of `{cash_ticker}` (residual cash earns `{cash_ticker}`; Sharpe/Sortino/PSR/DSR on excess returns)"
        if excess_enabled
        else "Raw net returns (residual cash earns 0.0%)"
    )
    sharpe_label = f"Sharpe (excess of {cash_ticker})" if excess_enabled else "Sharpe (net)"
    sortino_label = f"Sortino (excess of {cash_ticker})" if excess_enabled else "Sortino"

    lines = [
        f"# Champion Memo — `{record['strategy_name']}` ({record['cycle_id']})",
        "",
        f"- **Promoted at:** `{record['timestamp_utc']}` (trial #{record['trial_count']})",
        f"- **Policy SHA-256:** `{record['policy_sha256'][:16]}...` (verified immutable)",
        f"- **Dataset SHA-256:** `{record['dataset_sha256'][:16]}...` (development window only; holdout locked)",
        f"- **Evaluation Mode:** `{eval_mode}`",
        f"- **Scoring Convention:** {scoring_line}",
        f"- **{prior_line}**",
        f"- **Hypothesis:** {record['hypothesis']}",
        f"- **Parameters:** `{json.dumps(record['params'])}`",
    ]
    if dsr_val is not None:
        lines.append(
            f"- **Deflated Sharpe Ratio (DSR):** `{float(dsr_val):.4f}` (trial #{record['trial_count']}, {oos['observations']} OOS observations)"
        )
    if pdrop_val is not None:
        lines.append(f"- **Perturbation Stability:** worst OOS Sharpe drop `{float(pdrop_val):.6f}`")

    lines += [
        "",
        "## Purged Walk-Forward Out-of-Sample vs. SPY Benchmark (Net of 5 bps/side)",
        "",
        "| Metric | Candidate (OOS) | Candidate (IS) | SPY Buy & Hold (OOS) |",
        "| --- | ---: | ---: | ---: |",
        f"| CAGR | {oos['cagr']*100:.2f}% | {ins['cagr']*100:.2f}% | {bench['cagr']*100:.2f}% |",
        f"| Volatility | {oos['annual_volatility']*100:.2f}% | {ins['annual_volatility']*100:.2f}% | {bench['annual_volatility']*100:.2f}% |",
        f"| {sharpe_label} | {oos['sharpe']:.3f} | {ins['sharpe']:.3f} | {bench['sharpe']:.3f} |",
        f"| Raw Sharpe (total net) | {oos.get('raw_sharpe', oos['sharpe']):.3f} | {ins.get('raw_sharpe', ins['sharpe']):.3f} | {bench.get('raw_sharpe', bench['sharpe']):.3f} |",
        f"| {sortino_label} | {oos['sortino']:.3f} | {ins['sortino']:.3f} | {bench['sortino']:.3f} |",
        f"| Deflated Sharpe (N={record['trial_count']}) | {f'{float(dsr_val):.4f}' if dsr_val is not None else 'n/a'} | — | — |",
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


def run_holdout_audit(
    *,
    strategy_fn: StrategyFn,
    params: dict[str, Any],
    strategy_name: str,
    root_dir: str | Path = ".",
    output_path: Path | None = None,
    confirm_holdout_audit: bool = False,
    enforce_single_asset_cap: bool = True,
) -> dict[str, Any]:
    """Run a one-time sealed holdout audit ONLY when explicitly opted in via `confirm_holdout_audit=True`."""
    if not confirm_holdout_audit:
        raise PermissionError(
            "Holdout window is locked (splits.holdout_locked=true). "
            "Pass confirm_holdout_audit=True (or --confirm-holdout-audit) to run a one-time holdout audit."
        )

    root = Path(root_dir)
    policy = load_policy(root / "config" / "risk_policy.json")
    holdout_start = pd.Timestamp(policy.splits["holdout_start"])
    full_prices, manifest = load_frozen_prices(
        root / "data",
        tickers=policy.universe,
        verify_hash=True,
    )
    if full_prices.index[-1] < holdout_start:
        raise ValueError(
            f"Frozen dataset ends on {full_prices.index[-1].strftime('%Y-%m-%d')}, "
            f"before holdout_start {holdout_start.strftime('%Y-%m-%d')}."
        )

    cand_weights = strategy_fn(full_prices, params)
    bt_1x = run_backtest(
        full_prices,
        cand_weights,
        policy,
        enforce_single_asset_cap=enforce_single_asset_cap,
        cost_multiplier=1.0,
    )
    bt_3x = run_backtest(
        full_prices,
        cand_weights,
        policy,
        enforce_single_asset_cap=enforce_single_asset_cap,
        cost_multiplier=3.0,
    )
    bench_weights = _spy_benchmark_weights(full_prices, {"benchmark_ticker": policy.benchmark_ticker})
    bench_1x = run_backtest(
        full_prices,
        bench_weights,
        policy,
        enforce_single_asset_cap=False,
        cost_multiplier=1.0,
    )

    from engine.backtest import summarize_returns

    def _slice_holdout(bt: dict[str, Any]) -> dict[str, Any]:
        net_h = bt["net_returns"].loc[bt["net_returns"].index >= holdout_start]
        to_h = bt["turnover"].loc[bt["turnover"].index >= holdout_start]
        ex_h = (
            bt["excess_returns"].loc[bt["excess_returns"].index >= holdout_start]
            if bt["excess_returns"] is not None
            else None
        )
        return summarize_returns(
            net_h,
            to_h,
            rebalance_events=int((to_h > 1e-4).sum()),
            excess_returns=ex_h,
        )

    cand_h_1x = _slice_holdout(bt_1x)
    cand_h_3x = _slice_holdout(bt_3x)
    bench_h_1x = _slice_holdout(bench_1x)
    audit = {
        "audited_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_sha256": policy.sha256,
        "dataset_sha256": manifest["sha256"],
        "holdout_window": {
            "start": cand_h_1x["start_date"],
            "end": cand_h_1x["end_date"],
        },
        "candidate": {
            "name": strategy_name,
            "holdout_1x": cand_h_1x,
            "holdout_3x_sharpe": cand_h_3x["sharpe"],
        },
        "benchmark_spy": {
            "name": f"{policy.benchmark_ticker}_buy_and_hold",
            "holdout_1x": bench_h_1x,
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
