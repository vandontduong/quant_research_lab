"""Concise, evidence-backed reasoning trace generator for quant_research_lab experiments."""

from __future__ import annotations

from typing import Any


def generate_reasoning_trace(
    record: dict[str, Any],
    *,
    prior_champion: dict[str, Any] | None = None,
    complexity: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a compact, structured reasoning trace explaining why an experiment won, lost, or crashed."""
    decision = str(record.get("decision", "DISCARD")).upper()
    cycle_id = str(record.get("cycle_id", "unknown"))
    strat = str(record.get("strategy_name", "unknown"))

    if decision == "CRASH":
        err = str(record.get("error", "Unhandled exception during evaluation"))
        verdict = f"CRASH ({cycle_id} `{strat}`): {err}"
        next_hint = (
            "Fix the runtime/causality exception (inspect stack trace) before retrying, "
            "or discard the idea if the signal construction is structurally invalid."
        )
        return {
            "verdict": verdict,
            "hypothesis_check": "Not evaluated due to runtime/causality crash.",
            "performance_drivers": "No valid OOS return series produced.",
            "risk_and_overfit_audit": "Run aborted before gate evaluation.",
            "next_action_hint": next_hint,
            "summary_lines": [
                f"Verdict: {verdict}",
                f"Next: {next_hint}",
            ],
        }

    oos = record.get("out_of_sample") or {}
    ins = record.get("in_sample") or {}
    bench = record.get("benchmark_oos") or {}
    gates = record.get("gates") or {}
    failed_gates = list(gates.get("failed_gates") or [])
    attrib = record.get("attribution") or {}
    regimes = record.get("regime_breakdown") or {}
    costs = record.get("cost_sensitivity") or {}
    folds = record.get("folds") or []
    fc = record.get("fold_consistency") or {}
    dsr = record.get("deflated_sharpe")
    comp = complexity or record.get("complexity") or {}

    oos_sharpe = float(oos.get("sharpe", 0.0))
    is_sharpe = float(ins.get("sharpe", 0.0))
    oos_dd = float(oos.get("max_drawdown", 0.0))
    oos_to = float(oos.get("annual_turnover", 0.0))
    bench_sharpe = float(bench.get("sharpe", 0.0)) if "sharpe" in bench else None

    # 1. Verdict
    if prior_champion and prior_champion.get("out_of_sample"):
        ch_name = str(prior_champion.get("strategy_name", "champion"))
        ch_oos = prior_champion["out_of_sample"]
        ch_sharpe = float(ch_oos.get("sharpe", 0.0))
        ch_dd = float(ch_oos.get("max_drawdown", 0.0))
        ch_to = float(ch_oos.get("annual_turnover", 0.0))
        d_sharpe = oos_sharpe - ch_sharpe
        d_dd = oos_dd - ch_dd  # positive = shallower drawdown (better)
        d_to = oos_to - ch_to
        if decision == "PROMOTE":
            verdict = f"PROMOTE `{strat}`: OOS Sharpe {oos_sharpe:.3f} ({d_sharpe:+.3f} vs `{ch_name}`), all gates passed."
        else:
            fail_str = ", ".join(failed_gates) if failed_gates else "policy gate"
            verdict = (
                f"DISCARD `{strat}`: OOS Sharpe {oos_sharpe:.3f} ({d_sharpe:+.3f} vs `{ch_name}`); "
                f"failed [{fail_str}]."
            )
    else:
        ch_name = None
        d_sharpe = None
        d_dd = None
        d_to = None
        verdict = (
            f"{decision} `{strat}` (baseline): OOS Sharpe {oos_sharpe:.3f}, "
            f"Max DD {oos_dd * 100:.2f}%, Turnover {oos_to * 100:.1f}%."
        )

    # 2. Hypothesis check (evidence vs prior champion or SPY)
    hyp_parts: list[str] = []
    if d_sharpe is not None and d_dd is not None and d_to is not None:
        dd_word = "reduced" if d_dd > 0.005 else ("worsened" if d_dd < -0.005 else "held flat")
        to_word = "increased" if d_to > 0.05 else ("reduced" if d_to < -0.05 else "kept flat")
        hyp_parts.append(
            f"vs `{ch_name}`: Sharpe {d_sharpe:+.3f}, Max DD {dd_word} ({oos_dd * 100:.1f}% vs "
            f"{(oos_dd - d_dd) * 100:.1f}%), turnover {to_word} ({oos_to * 100:.0f}% vs {(oos_to - d_to) * 100:.0f}%)."
        )
    elif bench_sharpe is not None:
        hyp_parts.append(
            f"vs SPY OOS ({bench_sharpe:.3f}): spread {oos_sharpe - bench_sharpe:+.3f}, "
            f"Max DD {oos_dd * 100:.1f}%."
        )
    hypothesis_check = " ".join(hyp_parts) if hyp_parts else "Baseline established."

    # 3. Performance drivers (attribution + worst fold / stress regime)
    driver_parts: list[str] = []
    if attrib:
        top_c = attrib.get("top_contributor")
        bot_c = attrib.get("bottom_contributor")
        eff_bets = attrib.get("avg_effective_bets")
        cash_w = attrib.get("avg_cash_weight")
        beta = attrib.get("benchmark_beta")
        if top_c and bot_c:
            driver_parts.append(
                f"Top contributor `{top_c['ticker']}` ({top_c['annualized_return'] * 100:+.2f}%/yr), "
                f"lagging `{bot_c['ticker']}` ({bot_c['annualized_return'] * 100:+.2f}%/yr)."
            )
        if eff_bets is not None and cash_w is not None and beta is not None:
            driver_parts.append(
                f"Diversification {eff_bets:.2f} eff. bets, avg cash {cash_w * 100:.1f}%, SPY beta {beta:.2f}."
            )
    if regimes:
        worst_reg_name, worst_reg = min(regimes.items(), key=lambda kv: float(kv[1].get("total_return", 0.0)))
        driver_parts.append(
            f"Hardest stress regime `{worst_reg_name}` "
            f"(ret {float(worst_reg['total_return']) * 100:+.1f}%, DD {float(worst_reg['max_drawdown']) * 100:.1f}%)."
        )
    elif folds:
        worst_fold = min(folds, key=lambda f: float(f.get("sharpe", 0.0)))
        driver_parts.append(
            f"Weakest WF fold #{worst_fold['fold']} (`{worst_fold['start_date']}`..`{worst_fold['end_date']}`, "
            f"Sharpe {float(worst_fold['sharpe']):.2f}, DD {float(worst_fold['max_drawdown']) * 100:.1f}%)."
        )
    if fc:
        driver_parts.append(
            f"Fold Sharpe min/med/std={fc['min_fold_sharpe']:.2f}/{fc['median_fold_sharpe']:.2f}/"
            f"{fc['std_fold_sharpe']:.2f} ({fc['positive_fold_ratio'] * 100:.0f}% positive)."
        )
    performance_drivers = " ".join(driver_parts) if driver_parts else "No attribution diagnostics recorded."

    # 4. Risk, overfitting & simplicity audit
    audit_parts: list[str] = [f"IS→OOS gap {is_sharpe - oos_sharpe:+.3f}"]
    pdrop = record.get("perturbation_sharpe_drop")
    if pdrop is not None:
        audit_parts.append(f"perturbation drop {float(pdrop):+.3f}")
    if dsr is not None:
        audit_parts.append(f"DSR(n={record.get('trial_count', 1)})={float(dsr):.1%}")
    elif "psr_vs_zero" in oos:
        audit_parts.append(f"PSR(>0)={float(oos['psr_vs_zero']):.1%}")
    if "3x_15bps" in costs and "1x_5bps" in costs:
        drag = float(costs["1x_5bps"]["oos_sharpe"]) - float(costs["3x_15bps"]["oos_sharpe"])
        audit_parts.append(f"3x-cost Sharpe drag {drag:.3f}")
    if comp and comp.get("sloc"):
        audit_parts.append(f"complexity {comp.get('param_count', 0)} params / {comp['sloc']} SLOC")
    risk_and_overfit_audit = "; ".join(audit_parts) + "."

    # 5. Actionable next-step hint derived from observed evidence
    next_action_hint = _derive_next_action_hint(
        decision=decision,
        failed_gates=failed_gates,
        d_sharpe=d_sharpe,
        oos_to=oos_to,
        oos_dd=oos_dd,
        regimes=regimes,
        costs=costs,
    )

    summary_lines = [
        f"Verdict: {verdict}",
        f"Evidence: {hypothesis_check}",
        f"Drivers: {performance_drivers}",
        f"Audit: {risk_and_overfit_audit}",
        f"Next: {next_action_hint}",
    ]

    return {
        "verdict": verdict,
        "hypothesis_check": hypothesis_check,
        "performance_drivers": performance_drivers,
        "risk_and_overfit_audit": risk_and_overfit_audit,
        "next_action_hint": next_action_hint,
        "summary_lines": summary_lines,
    }


def _derive_next_action_hint(
    *,
    decision: str,
    failed_gates: list[str],
    d_sharpe: float | None,
    oos_to: float,
    oos_dd: float,
    regimes: dict[str, Any],
    costs: dict[str, Any],
) -> str:
    if "max_oos_drawdown" in failed_gates:
        return (
            f"Drawdown ({oos_dd * 100:.1f}%) breached policy ceiling; add volatility scaling or "
            "defensive-sleeve floor rather than concentrating into high-beta momentum."
        )
    if "perturbation_stability" in failed_gates or "is_to_oos_sharpe_degradation" in failed_gates:
        return (
            "Overfitting/parameter-cliff detected; simplify signal thresholds, widen lookback windows, "
            "and cut tunable parameter count."
        )
    if "max_annual_turnover" in failed_gates or (
        "3x_15bps" in costs
        and "1x_5bps" in costs
        and float(costs["1x_5bps"]["oos_sharpe"]) - float(costs["3x_15bps"]["oos_sharpe"]) > 0.08
    ):
        return (
            f"Turnover ({oos_to * 100:.0f}%/yr) creates material cost drag; add rebalance deadband/hysteresis "
            "or slow the signal cadence."
        )
    if decision == "DISCARD" and failed_gates == ["beats_champion_margin"]:
        if d_sharpe is not None and d_sharpe > 0.0:
            return (
                f"Positive but sub-margin gain ({d_sharpe:+.3f} < +0.050); test whether reducing turnover "
                f"({oos_to * 100:.0f}%/yr) or combining with a low-correlation defensive filter clears the bar."
            )
        if d_sharpe is not None and d_sharpe < -0.05:
            return (
                "Signal materially underperformed champion (whipsaw or cash drag); avoid hard binary "
                "cutoffs and test continuous risk-budget adjustments instead."
            )
    if regimes:
        worst_name, worst_m = min(regimes.items(), key=lambda kv: float(kv[1].get("total_return", 0.0)))
        if float(worst_m.get("total_return", 0.0)) < -0.08:
            return (
                f"Target `{worst_name}` drawdown ({float(worst_m['max_drawdown']) * 100:.1f}%) "
                "using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation."
            )
    return "Target the weakest walk-forward fold while keeping parameter count and turnover low."


def format_reasoning_trace_markdown(record: dict[str, Any]) -> str:
    """Format a record's reasoning_trace as a compact Markdown section."""
    trace = record.get("reasoning_trace")
    if not isinstance(trace, dict) or not trace.get("summary_lines"):
        trace = generate_reasoning_trace(record)
    cycle_id = record.get("cycle_id", "unknown")
    strat = record.get("strategy_name", "unknown")
    decision = record.get("decision", "DISCARD")
    lines = [f"### `{cycle_id}` — `{strat}` (`{decision}`)", ""]
    for item in trace["summary_lines"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
