"""Autoresearch-style tab-separated results.tsv exporter and crash record logger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.runner import get_short_git_commit, read_ledger
from tools.complexity import analyze_strategy_complexity

TSV_HEADER = (
    "commit\tcycle_id\tstrategy\tstrategy_note\toos_sharpe\toos_max_dd\t"
    "annual_turnover\tsloc\tparams\tstatus\tdescription"
)

STRATEGY_NOTES: dict[str, str] = {
    "v000_equal_weight_monthly": "Monthly equal-weight (1/6) across all 6 ETFs (naive baseline).",
    "v001_inv_vol_monthly": "Monthly 63d inverse-volatility weighting with 4% vol floor and 45% cap.",
    "v002_inv_vol_trend_filter": "Inverse-vol with binary 126d SMA trend cutoff routing below-SMA assets to cash.",
    "v003_inv_vol_top_momentum": "Inverse-vol concentrated in top-3 ETFs ranked by trailing 126d momentum.",
    "v004_corr_aware_risk_parity": "Inverse-vol penalized by average positive 63d pairwise correlation.",
    "v005_soft_trend_inv_vol": "Inverse-vol tilted smoothly by exp(0.5 * z_excess_return vs SHY), 100% invested.",
    "v006_vol_targeted_inv_vol": "Inverse-vol risky sleeve scaled to 8% fast 21d portfolio vol target; excess to SHY/cash.",
    "v007_erc_risk_parity": "True Equal Risk Contribution (ERC) solved on shrunk 126d covariance matrix.",
    "v008_downside_vol_inv_weight": "Inverse-risk weighting blending 60% downside semideviation and 40% total vol.",
    "v009_rate_regime_inv_vol": "Inverse-vol scaling down TLT when trailing 63d SPY-TLT correlation turns positive.",
    "v010_dmn_sharpe_opt": "DMN (arXiv:1904.04912): direct Sharpe-ratio optimization over multi-horizon vol-scaled trends.",
    "v011_momentum_transformer_cpd": "Momentum Transformer + CPD (arXiv:2112.08534): 4-horizon attention dampened by changepoint severity.",
    "v012_drl_vol_scaled_policy": "DRL Trading (arXiv:1911.10107): causal online policy-gradient agent on vol-scaled net rewards.",
    "v013_sdf_macro_factor": "Deep Asset Pricing / SDF (arXiv:1904.00745): macro-state conditioned regularized GMM SDF tilts.",
    "v014_stat_arb_residual_mr": "Deep StatArb (arXiv:2106.04028): PCA-orthogonal residual multi-scale mean reversion + cost deadband.",
    "v015_eiie_portfolio_rl": "EIIE Portfolio RL (arXiv:1706.10059): shared per-asset evaluator + prev-weight cost-remainder softmax.",
    "v016_mm_deeplob_inventory": "MM-RL + DeepLOB (arXiv:1804.04216 / 1808.03668): multi-scale inception pressure + inventory dampening.",
    "v017_news_underreaction_pead": "LLM News Sentiment proxy (arXiv:2304.07619): decaying post-shock underreaction drift tilts.",
    "v018_trading_agents_debate": "TradingAgents (arXiv:2412.20138): 4-role debate (trend, MR, macro, risk-manager disagreement veto).",
    "v019_qlib_factor_ensemble": "Qlib workflow (arXiv:2009.11189): causal rolling-Sharpe softmax ensemble over 4 risk-parity models.",
}


def get_strategy_note(record: dict[str, Any]) -> str:
    """Return a concise single-line note explaining a strategy."""
    explicit = str(record.get("strategy_note") or "").strip()
    if explicit:
        return _sanitize_tsv_field(explicit, max_len=110)
    strat_name = str(record.get("strategy_name") or "").strip()
    if strat_name in STRATEGY_NOTES:
        return STRATEGY_NOTES[strat_name]
    hyp = str(record.get("hypothesis") or "").strip()
    if hyp:
        first_sentence = hyp.split(". ")[0].rstrip(".") + "."
        return _sanitize_tsv_field(first_sentence, max_len=110)
    return "Candidate strategy."


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
    raw_commit = str(record.get("git_commit") or "").strip()
    commit = _sanitize_tsv_field(raw_commit or "unknown", max_len=32)
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

    strat_name = _sanitize_tsv_field(str(record.get("strategy_name", "unknown")), max_len=48)
    strat_note = get_strategy_note(record)
    hyp = str(record.get("hypothesis", "")).strip()
    if status == "crash" and record.get("error"):
        desc_raw = f"CRASH ({record['error']})"
    else:
        desc_raw = hyp or strat_note
    desc = _sanitize_tsv_field(desc_raw, max_len=140)

    return (
        f"{commit}\t{cycle_id}\t{strat_name}\t{strat_note}\t{oos_sharpe:.6f}\t"
        f"{oos_max_dd:.6f}\t{ann_to:.4f}\t{sloc}\t{param_count}\t{status}\t{desc}"
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
