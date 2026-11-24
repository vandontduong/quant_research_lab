# Autoresearch Playbook for `quant_research_lab`

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (`program.md`), hardened with `quant_research_lab`'s immutable risk policy, prefix-invariance causality checks, purged walk-forward validation, and sealed `2024+` holdout.

> **Research-only boundary:** Offline historical simulation only — not investment advice, no live brokerage integrations, and no order-routing code.

## Setup

1. **Agree on a run tag** (e.g. `nov23`) and create a dedicated experiment branch:
   ```bash
   git checkout -b autoresearch/<tag>
   ```
2. **Read the in-scope files**:
   - `PROMPT.md` — full scientific guardrails and cycle rules.
   - `config/risk_policy.json` — immutable risk ceilings and promotion gates (**read-only**).
   - `experiments/best_candidate/MEMO.md` — current champion metrics and observed weaknesses.
   - `experiments/results.tsv` — compact tab-separated experiment log.
3. **Verify harness integrity**:
   ```bash
   uv sync --extra dev
   uv run pytest -v
   uv run ruff check src tests scripts
   ```

## Scope Boundaries

**What you CAN edit:**
- Add or iterate on candidate strategy modules under `src/strategies/` (strictly causal, deterministic, parameterized via `DEFAULT_PARAMS`).

**What you CANNOT edit during optimization:**
- `config/risk_policy.json` (SHA-256 verified on every cycle; never weaken risk gates or promotion margins).
- `data/` (frozen total-return ETF dataset + `manifest.json`).
- `src/engine/` (fixed backtester, purged walk-forward evaluator, and locked `2024+` holdout enforcement).
- Dependencies in `pyproject.toml` (only use installed packages).

Use `--check-git-clean-harness` on `scripts/run_experiment.py` to fail fast if any immutable harness file was modified.

## Objective & Simplicity Criterion

- **Primary objective:** Maximize purged walk-forward **out-of-sample excess Sharpe vs. `SHY`** (`oos_sharpe`, with residual cash earning `SHY`) on the development window (`2005-01-03` .. `2023-12-29`) by at least `+0.05` over the current champion, while passing all hard risk gates (drawdown, turnover, rebalance count, IS→OOS degradation, perturbation stability, and `min_deflated_sharpe >= 0.50`).
- **Evaluation modes:**
  - `fixed_param_causal_oos`: default for fixed-rule strategies (`generate_weights(prices, params)` only); runs one causal weight pass and slices into 21-day-purged test folds.
  - `per_fold_purged_wf`: enabled automatically when a strategy module defines `fit_strategy(train_prices, params)` or `fit_params(train_prices, params)`; calls the fitter strictly on each fold's training window (`train_end + 21 <= test_start`) before causal test-fold weight generation.
- **Simplicity criterion (from `autoresearch`):** All else being equal, simpler strategies generalize better and carry lower overfitting risk. Every run logs `param_count`, `sloc`, and `branch_count`. Prefer fewer parameters and cleaner logic over fragile multi-knob heuristics; a simplification that matches or improves `oos_sharpe` while cutting `param_count`/`sloc` is a strong win.

## Single-Command Experiment Loop

Run bounded cycles (up to `promotion.max_cycles_per_invocation` and stopping after `promotion.stop_after_consecutive_discards` consecutive discards):

1. **Inspect state:** Check `experiments/best_candidate/MEMO.md` and `uv run scripts/analyze_experiments.py` to identify the current champion's biggest weakness and which gates rejected prior attempts.
2. **Implement one falsifiable hypothesis** in a new `src/strategies/vNNN_<name>.py` module exposing `STRATEGY_NAME`, `DEFAULT_PARAMS`, `HYPOTHESIS`, `generate_weights(prices, params)`, and optional `NEIGHBOR_PARAMS`.
3. **Run the experiment** with output redirected to `run.log` (keeping agent context clean):
   ```bash
   uv run scripts/run_experiment.py \
     --strategy src/strategies/vNNN_<name>.py \
     --timeout-sec 120 \
     --check-git-clean-harness > run.log 2>&1
   ```
   (Add `--dry-run` first if you want to verify causality and metrics without appending to `experiments/ledger.jsonl`.)
4. **Read the grep-friendly summary block:**
   ```bash
   grep "^oos_sharpe:\|^oos_max_drawdown:\|^param_count:\|^sloc:\|^decision:\|^failed_gates:\|^error:" run.log
   ```
   Example output:
   ```
   ---
   cycle_id:            cycle_007
   strategy_name:       v007_example
   git_commit:          be7650f
   dry_run:             false
   oos_sharpe:          1.095120
   is_sharpe:           1.062340
   oos_cagr:            0.071200
   oos_max_drawdown:    -0.142100
   annual_turnover:     0.421000
   param_count:         4
   sloc:                58
   branch_count:        7
   runtime_sec:         0.482
   status:              keep
   decision:            PROMOTE
   failed_gates:        none
   ```
5. **Keep, discard, or crash:**
   - `PROMOTE` (`keep`): `experiments/ledger.jsonl`, `experiments/results.tsv`, and `experiments/best_candidate/` are updated automatically. Commit the winning strategy on your `autoresearch/<tag>` branch.
   - `DISCARD` (`discard`): The full record is still appended to `experiments/ledger.jsonl` and `experiments/results.tsv` so future cycles avoid repeating dead ends.
   - `CRASH` (`crash`): Exceptions and timeouts are caught and logged as `CRASH` in `experiments/ledger.jsonl` and `experiments/results.tsv` (`oos_sharpe = 0.000000`). Inspect `tail -n 50 run.log`, fix trivial bugs, or move on.
6. **Refresh analytics & frontier chart:**
   ```bash
   uv run scripts/analyze_experiments.py
   ```
   Updates `experiments/results.tsv` and renders `experiments/progress.svg` (running-best OOS Sharpe frontier vs `SPY` benchmark, promoted champions, discarded trials, and gate failure breakdown).
