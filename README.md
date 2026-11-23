# quant_research_lab

Offline recursive quantitative research and purged walk-forward backtesting sandbox (Python 3.11+, `uv`).

> **Research-only boundary:** Offline historical simulation only — not investment advice, no live brokerage integrations, and no order-routing code.

## Layout

- `PROMPT.md` — Hardened recursive research loop specification and guardrails.
- `AUTORESEARCH.md` — Autonomous experiment playbook adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (`program.md`), preserving `quant_research_lab`'s immutable policy, causality checks, and locked holdout.
- `config/risk_policy.json` — Immutable risk ceilings, execution assumptions, survivorship-safe ETF universe (`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `SHY`), purged walk-forward splits, and locked `2024+` holdout boundary (SHA-256 verified on every cycle).
- `data/` — Frozen total-return (`auto_adjust=True`) daily ETF dataset (`etf_daily_total_return.csv`) and `manifest.json` with SHA-256 provenance.
- `src/engine/` — Deterministic next-bar execution backtester (`backtest.py`), immutable policy & gate checker (`policy.py`), purged walk-forward + prefix-invariance lookahead + perturbation + stress-regime & cost-sensitivity diagnostics (`validation.py`), and cycle runner (`runner.py`).
- `src/strategies/` — Isolated, deterministic, strictly causal candidate strategies (`v000_baseline.py` through `v006_vol_targeted_inv_vol.py`).
- `src/tools/` & `scripts/` — Autoresearch-inspired experiment tooling: single-command runner with timeout, crash logging, and grep-friendly summary (`scripts/run_experiment.py`), AST/parameter simplicity auditor (`src/tools/complexity.py`), tab-separated `results.tsv` exporter (`src/tools/results_tsv.py`), and progress analyzer + zero-dependency SVG frontier chart (`scripts/analyze_experiments.py`).
- `experiments/` — Append-only experiment log (`ledger.jsonl`), compact tab-separated view (`results.tsv`), OOS Sharpe frontier chart (`progress.svg`), one-time sealed holdout audit (`holdout_audit.json`), and promoted champion snapshot (`best_candidate/`).
- `tests/` — Offline rigor tests verifying causality/no-lookahead, NaN/Inf rejection, leverage/concentration caps, locked holdout enforcement, cost-sensitivity monotonicity, and experiment tooling.

## Quickstart

```bash
uv sync --extra dev
uv run ruff check src tests scripts
uv run pytest -v

# Analyze experiment history, sync experiments/results.tsv, and render experiments/progress.svg
uv run scripts/analyze_experiments.py

# Dry-run a candidate strategy with autoresearch-style grep-friendly output
uv run scripts/run_experiment.py --strategy src/strategies/v001_inv_vol.py --dry-run
```
