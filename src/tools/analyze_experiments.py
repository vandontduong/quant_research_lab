"""Autoresearch-style experiment analyzer & zero-dependency SVG frontier chart generator."""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from engine.policy import load_policy
from engine.reasoning import format_reasoning_trace_markdown, generate_reasoning_trace
from engine.runner import read_ledger
from tools.complexity import analyze_strategy_complexity
from tools.results_tsv import sync_results_tsv


def _status_of(record: dict[str, Any]) -> str:
    d = str(record.get("decision", "DISCARD")).strip().upper()
    if d in {"PROMOTE", "KEEP"}:
        return "KEEP"
    if d == "CRASH":
        return "CRASH"
    return "DISCARD"


def compute_experiment_summary(
    rows: list[dict[str, Any]],
    *,
    root_dir: str | Path = ".",
) -> dict[str, Any]:
    """Compute autoresearch-style summary stats, frontier progression, top hits, and gate failures."""
    root = Path(root_dir)
    counts = Counter(_status_of(r) for r in rows)
    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_crash = counts.get("CRASH", 0)
    n_decided = n_keep + n_discard
    keep_rate = (n_keep / n_decided) if n_decided > 0 else 0.0

    # Trailing consecutive non-keep runs (useful against stop_after_consecutive_discards).
    consecutive_discards = 0
    for r in reversed(rows):
        if _status_of(r) == "KEEP":
            break
        consecutive_discards += 1

    kept_entries: list[dict[str, Any]] = []
    gate_failures: Counter[str] = Counter()
    benchmark_sharpe: float | None = None

    for idx, r in enumerate(rows):
        status = _status_of(r)
        bench = r.get("benchmark_oos") or {}
        if benchmark_sharpe is None and "sharpe" in bench:
            benchmark_sharpe = float(bench["sharpe"])

        if status == "DISCARD":
            for g in (r.get("gates") or {}).get("failed_gates", []):
                gate_failures[str(g)] += 1

        if status == "KEEP":
            oos = r.get("out_of_sample") or {}
            comp = r.get("complexity")
            if not isinstance(comp, dict):
                strat_rel = r.get("strategy_file", "")
                comp = analyze_strategy_complexity(
                    (root / strat_rel) if strat_rel else Path(),
                    r.get("params"),
                )
            kept_entries.append(
                {
                    "index": idx,
                    "cycle_id": str(r.get("cycle_id", f"cycle_{idx:03d}")),
                    "strategy_name": str(r.get("strategy_name", "unknown")),
                    "oos_sharpe": float(oos.get("sharpe", 0.0)),
                    "oos_max_dd": float(oos.get("max_drawdown", 0.0)),
                    "annual_turnover": float(oos.get("annual_turnover", 0.0)),
                    "sloc": int(comp.get("sloc", 0)),
                    "param_count": int(comp.get("param_count", 0)),
                    "hypothesis": str(r.get("hypothesis", "")),
                }
            )

    top_hits: list[dict[str, Any]] = []
    for i in range(1, len(kept_entries)):
        prev = kept_entries[i - 1]
        cur = kept_entries[i]
        top_hits.append(
            {
                **cur,
                "delta_sharpe": round(cur["oos_sharpe"] - prev["oos_sharpe"], 6),
                "delta_dd": round(cur["oos_max_dd"] - prev["oos_max_dd"], 6),
                "delta_sloc": cur["sloc"] - prev["sloc"],
            }
        )
    top_hits.sort(key=lambda x: x["delta_sharpe"], reverse=True)

    baseline_sharpe = kept_entries[0]["oos_sharpe"] if kept_entries else None
    best_entry = max(kept_entries, key=lambda x: x["oos_sharpe"]) if kept_entries else None
    total_improvement = (
        round(best_entry["oos_sharpe"] - baseline_sharpe, 6)
        if (best_entry is not None and baseline_sharpe is not None)
        else 0.0
    )

    return {
        "total_experiments": len(rows),
        "n_keep": n_keep,
        "n_discard": n_discard,
        "n_crash": n_crash,
        "keep_rate": keep_rate,
        "consecutive_discards": consecutive_discards,
        "benchmark_oos_sharpe": benchmark_sharpe,
        "baseline_oos_sharpe": baseline_sharpe,
        "best_entry": best_entry,
        "total_improvement": total_improvement,
        "kept_entries": kept_entries,
        "top_hits": top_hits,
        "gate_failures": dict(gate_failures.most_common()),
    }


def render_progress_svg(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Render an autoresearch-style SVG progress chart (higher OOS Sharpe is better)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    width, height = 980, 480
    pad_l, pad_r, pad_t, pad_b = 78, 42, 56, 68
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    valid_points: list[tuple[int, float, str, str]] = []
    crash_indices: list[int] = []
    benchmark_sharpe: float | None = None

    for idx, r in enumerate(rows):
        status = _status_of(r)
        bench = r.get("benchmark_oos") or {}
        if benchmark_sharpe is None and "sharpe" in bench:
            benchmark_sharpe = float(bench["sharpe"])
        if status == "CRASH":
            crash_indices.append(idx)
            continue
        oos = r.get("out_of_sample") or {}
        if "sharpe" in oos:
            valid_points.append(
                (
                    idx,
                    float(oos["sharpe"]),
                    status,
                    str(r.get("strategy_name", r.get("cycle_id", f"#{idx}"))),
                )
            )

    n_total = len(rows)
    n_kept = sum(1 for _, _, st, _ in valid_points if st == "KEEP")

    y_vals = [y for _, y, _, _ in valid_points]
    if benchmark_sharpe is not None:
        y_vals.append(benchmark_sharpe)
    if not y_vals:
        y_vals = [0.0, 1.0]

    y_min_raw = min(y_vals)
    y_max_raw = max(y_vals)
    span = max(y_max_raw - y_min_raw, 0.25)
    y_min = y_min_raw - span * 0.18
    y_max = y_max_raw + span * 0.22

    max_x = max(n_total - 1, 1)

    def sx(idx: float) -> float:
        return pad_l + (idx / max_x) * plot_w

    def sy(val: float) -> float:
        return pad_t + (1.0 - (val - y_min) / (y_max - y_min)) * plot_h

    svg: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">'
        ),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
            'fill="#fafafa" stroke="#e5e7eb" stroke-width="1"/>'
        ),
    ]

    # Horizontal gridlines & Y ticks
    n_yticks = 5
    for i in range(n_yticks + 1):
        v = y_min + (y_max - y_min) * (i / n_yticks)
        yy = sy(v)
        svg.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
            'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'<text x="{pad_l - 10}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#4b5563">{v:.2f}</text>'
        )

    # X ticks
    step_x = max(1, n_total // 10)
    for idx in range(0, max(n_total, 1), step_x):
        xx = sx(idx)
        svg.append(
            f'<line x1="{xx:.1f}" y1="{pad_t + plot_h}" x2="{xx:.1f}" y2="{pad_t + plot_h + 5}" '
            'stroke="#9ca3af" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{xx:.1f}" y="{pad_t + plot_h + 20}" text-anchor="middle" '
            f'font-size="11" fill="#4b5563">{idx}</text>'
        )

    # Benchmark reference line (SPY OOS Sharpe)
    if benchmark_sharpe is not None:
        by = sy(benchmark_sharpe)
        svg.append(
            f'<line x1="{pad_l}" y1="{by:.1f}" x2="{width - pad_r}" y2="{by:.1f}" '
            'stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.8"/>'
        )
        svg.append(
            f'<text x="{width - pad_r - 6}" y="{by - 6:.1f}" text-anchor="end" '
            f'font-size="11" fill="#2563eb">SPY OOS Sharpe ({benchmark_sharpe:.3f})</text>'
        )

    # Discarded scatter points
    for idx, yv, st, _name in valid_points:
        if st != "KEEP":
            svg.append(
                f'<circle cx="{sx(idx):.1f}" cy="{sy(yv):.1f}" r="4.5" '
                'fill="#9ca3af" opacity="0.65"/>'
            )

    # Crash markers along bottom
    for idx in crash_indices:
        cx = sx(idx)
        cy = pad_t + plot_h - 10
        svg.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            'font-size="13" font-weight="700" fill="#dc2626">×</text>'
        )

    # Running-best frontier step line (kept points)
    kept_pts = [(idx, yv, name) for idx, yv, st, name in valid_points if st == "KEEP"]
    if kept_pts:
        path_cmds: list[str] = []
        running_best = -1e18
        prev_y_px: float | None = None
        for i, (idx, yv, _name) in enumerate(kept_pts):
            running_best = max(running_best, yv)
            xx = sx(idx)
            yy = sy(running_best)
            if i == 0:
                path_cmds.append(f"M {xx:.1f} {yy:.1f}")
            else:
                assert prev_y_px is not None
                path_cmds.append(f"L {xx:.1f} {prev_y_px:.1f}")
                path_cmds.append(f"L {xx:.1f} {yy:.1f}")
            prev_y_px = yy
        # Extend frontier horizontally to last experiment
        if n_total > 1 and kept_pts[-1][0] < n_total - 1 and prev_y_px is not None:
            path_cmds.append(f"L {sx(n_total - 1):.1f} {prev_y_px:.1f}")
        svg.append(
            f'<path d="{" ".join(path_cmds)}" fill="none" stroke="#16a34a" '
            'stroke-width="2.2" opacity="0.85"/>'
        )

    # Kept points & annotations
    for idx, yv, name in kept_pts:
        xx = sx(idx)
        yy = sy(yv)
        svg.append(
            f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6.5" '
            'fill="#22c55e" stroke="#14532d" stroke-width="1.5"/>'
        )
        short_label = name if len(name) <= 28 else name[:25] + "..."
        label = escape(f"{short_label} ({yv:.3f})")
        anchor = "end" if xx > (pad_l + plot_w * 0.72) else "start"
        dx = -10 if anchor == "end" else 10
        svg.append(
            f'<text x="{xx + dx:.1f}" y="{yy - 10:.1f}" text-anchor="{anchor}" '
            f'font-size="11" font-weight="600" fill="#166534">{label}</text>'
        )

    # Titles & axis labels
    title = escape(
        f"Quant Research Progress: {n_total} Experiments, {n_kept} Promoted Champions"
    )
    svg.append(
        f'<text x="{pad_l}" y="30" font-size="16" font-weight="700" fill="#111827">{title}</text>'
    )
    svg.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" '
        'font-size="12" fill="#374151">Experiment # (Trial Index)</text>'
    )
    svg.append(
        f'<text x="22" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-size="12" '
        f'fill="#374151" transform="rotate(-90 22 {pad_t + plot_h / 2:.1f})">'
        'Walk-Forward OOS Sharpe (higher is better)</text>'
    )

    # Compact legend in top-right
    leg_x = width - pad_r - 275
    leg_y = 22
    svg.extend(
        [
            f'<circle cx="{leg_x}" cy="{leg_y}" r="5" fill="#22c55e" stroke="#14532d" stroke-width="1.2"/>',
            f'<text x="{leg_x + 10}" y="{leg_y + 4}" font-size="11" fill="#374151">Promoted</text>',
            f'<circle cx="{leg_x + 82}" cy="{leg_y}" r="4.5" fill="#9ca3af" opacity="0.7"/>',
            f'<text x="{leg_x + 92}" y="{leg_y + 4}" font-size="11" fill="#374151">Discarded</text>',
            (
                f'<line x1="{leg_x + 162}" y1="{leg_y}" x2="{leg_x + 182}" y2="{leg_y}" '
                'stroke="#16a34a" stroke-width="2.2"/>'
            ),
            f'<text x="{leg_x + 188}" y="{leg_y + 4}" font-size="11" fill="#374151">Frontier</text>',
        ]
    )

    svg.append("</svg>")
    out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return out


def format_terminal_report(
    summary: dict[str, Any],
    *,
    stop_after_consecutive_discards: int | None = None,
) -> str:
    """Format an autoresearch-style terminal analysis report."""
    lines = [
        "=== Quant Research Lab — Experiment Analysis ===",
        (
            f"Total experiments: {summary['total_experiments']}  "
            f"(KEEP={summary['n_keep']}, DISCARD={summary['n_discard']}, CRASH={summary['n_crash']})"
        ),
        f"Keep rate:         {summary['keep_rate']:.1%}",
    ]
    if stop_after_consecutive_discards is not None:
        lines.append(
            f"Trailing discards: {summary['consecutive_discards']} / "
            f"{stop_after_consecutive_discards} (policy stop threshold)"
        )

    if summary["baseline_oos_sharpe"] is not None and summary["best_entry"] is not None:
        best = summary["best_entry"]
        lines.extend(
            [
                "",
                f"Baseline OOS Sharpe: {summary['baseline_oos_sharpe']:.4f}",
                f"Best OOS Sharpe:     {best['oos_sharpe']:.4f} ({best['strategy_name']}, {best['cycle_id']})",
                f"Total improvement:   {summary['total_improvement']:+.4f}",
            ]
        )
        if summary["benchmark_oos_sharpe"] is not None:
            lines.append(
                f"SPY OOS Sharpe:      {summary['benchmark_oos_sharpe']:.4f} "
                f"(spread: {best['oos_sharpe'] - summary['benchmark_oos_sharpe']:+.4f})"
            )

    kept = summary["kept_entries"]
    if kept:
        lines.extend(
            [
                "",
                f"Promoted Champions ({len(kept)}):",
                f"  {'#':>3}  {'Cycle':<10}  {'Sharpe':>7}  {'MaxDD':>8}  {'Turn':>6}  {'SLOC':>4}  {'Prm':>3}  Strategy",
                "  " + "-" * 78,
            ]
        )
        for k in kept:
            lines.append(
                f"  {k['index']:3d}  {k['cycle_id']:<10}  {k['oos_sharpe']:7.4f}  "
                f"{k['oos_max_dd']*100:7.2f}%  {k['annual_turnover']*100:5.1f}%  "
                f"{k['sloc']:4d}  {k['param_count']:3d}  {k['strategy_name']}"
            )

    hits = summary["top_hits"]
    if hits:
        lines.extend(
            [
                "",
                "Top Hits (ranked by incremental OOS Sharpe delta):",
                f"  {'Rank':>4}  {'Delta':>8}  {'Sharpe':>8}  {'dSLOC':>6}  Strategy",
                "  " + "-" * 64,
            ]
        )
        for rank, h in enumerate(hits, 1):
            lines.append(
                f"  {rank:4d}  {h['delta_sharpe']:+8.4f}  {h['oos_sharpe']:8.4f}  "
                f"{h['delta_sloc']:+6d}  {h['strategy_name']}"
            )

    gate_fails = summary["gate_failures"]
    if gate_fails:
        lines.extend(["", "Discard Gate Breakdown:"])
        for gate_name, count in gate_fails.items():
            lines.append(f"  - {gate_name}: {count}")

    return "\n".join(lines)


def sync_reasoning_traces_md(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    root_dir: str | Path = ".",
) -> int:
    """Write concise, chronological reasoning traces for all experiments to Markdown."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root_dir)
    chunks = [
        "# Experiment Reasoning Traces\n",
        "> Concise, evidence-backed traces auto-generated from `experiments/ledger.jsonl`.\n\n",
    ]
    running_champion: dict[str, Any] | None = None
    for row in rows:
        rec = dict(row)
        if not isinstance(rec.get("reasoning_trace"), dict) or not rec["reasoning_trace"].get("summary_lines"):
            strat_rel = rec.get("strategy_file", "")
            comp = analyze_strategy_complexity(
                (root / strat_rel) if strat_rel else Path(),
                rec.get("params"),
            )
            rec["reasoning_trace"] = generate_reasoning_trace(
                rec,
                prior_champion=running_champion,
                complexity=comp,
            )
        chunks.append(format_reasoning_trace_markdown(rec))
        if _status_of(rec) == "KEEP":
            running_champion = rec

    out.write_text("".join(chunks), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze quant_research_lab experiments, sync results.tsv, traces, and plot progress.svg."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory.")
    parser.add_argument(
        "--no-svg", action="store_true", help="Skip generating experiments/progress.svg."
    )
    parser.add_argument(
        "--no-tsv", action="store_true", help="Skip regenerating experiments/results.tsv."
    )
    parser.add_argument(
        "--no-traces", action="store_true", help="Skip regenerating experiments/reasoning_traces.md."
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    ledger_path = root / "experiments" / "ledger.jsonl"
    tsv_path = root / "experiments" / "results.tsv"
    svg_path = root / "experiments" / "progress.svg"
    traces_path = root / "experiments" / "reasoning_traces.md"

    rows = read_ledger(ledger_path)
    stop_limit: int | None = None
    policy_path = root / "config" / "risk_policy.json"
    if policy_path.exists():
        policy = load_policy(policy_path)
        stop_limit = int(policy.promotion.get("stop_after_consecutive_discards", 3))

    if not args.no_tsv:
        n_rows = sync_results_tsv(ledger_path, tsv_path, root_dir=root)
        print(f"Synced {n_rows} rows -> {tsv_path.relative_to(root)}")

    if not args.no_traces:
        n_traces = sync_reasoning_traces_md(rows, traces_path, root_dir=root)
        print(f"Synced {n_traces} reasoning traces -> {traces_path.relative_to(root)}")

    if not args.no_svg:
        render_progress_svg(rows, svg_path)
        print(f"Rendered frontier chart -> {svg_path.relative_to(root)}")

    summary = compute_experiment_summary(rows, root_dir=root)
    print()
    print(format_terminal_report(summary, stop_after_consecutive_discards=stop_limit))
    if rows:
        last_row = dict(rows[-1])
        running_champ = None
        for r in rows[:-1]:
            if _status_of(r) == "KEEP":
                running_champ = r
        trace = last_row.get("reasoning_trace") or generate_reasoning_trace(
            last_row, prior_champion=running_champ
        )
        print("\nLatest Experiment Reasoning Trace:")
        for line in trace.get("summary_lines", []):
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
