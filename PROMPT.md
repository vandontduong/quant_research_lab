# Recursive Self-Improvement Quantitative Research & Strategy System (v2 — Fixed & Hardened)

You are building and running an **offline Recursive Self-Improvement Quantitative Research & Strategy System** inside `/home/antigravity/quant_research_lab/` (isolated from any other repository in the workspace). Treat this as a disciplined quantitative research and software engineering loop—not a one-shot script—and enforce strict scientific rigor, anti-overfitting controls, and immutable risk guardrails on every iteration.

> **Research-only boundary:** Offline historical simulation and quantitative research only. Never add brokerage integrations, live order routing, real-capital execution, or buy/sell/position-size advice.

---

### 1. Workspace & Persistent State Layout (`/home/antigravity/quant_research_lab/`)
Maintain a clean, modular structure:
- `config/risk_policy.json` — **READ-ONLY by the self-improvement loop.** Contains hard risk ceilings, execution lag, slippage/commission assumptions, survivorship-safe ETF universe, walk-forward + locked holdout splits, and promotion thresholds. Programmatically verified via SHA-256 hash on every cycle; never weaken these constraints during optimization.
- `data/` — Frozen, versioned historical total-return (`auto_adjust=True`) market datasets with `manifest.json` recording exact tickers, date ranges, corporate-action basis, and file SHA-256 hashes so every cycle is 100% reproducible.
- `src/engine/` — Deterministic backtesting & purged walk-forward evaluation harness:
  - Enforces **strict causality**: signal at close `t` executes with at least a 1-bar lag (`t -> t+1`), verified by an automated prefix-invariance (no-lookahead) test.
  - Simulates realistic one-way turnover accounting for natural weight drift between rebalances, plus commission + slippage bps.
  - Enforces a **locked final holdout window** (`holdout_start` onward) that is strictly excluded from the recursive loop to prevent multi-cycle OOS overfitting.
  - Logs `trial_count` (multiple-testing tracker) and checks parameter-neighborhood stability (perturbation test).
- `src/strategies/` — Candidate strategy implementations (`v000_baseline.py`, `v001_*.py`, `v002_*.py`, ...), each isolated, deterministic, and parameterized cleanly.
- `experiments/ledger.jsonl` — Append-only JSONL log of every cycle: `cycle_id`, `trial_count`, `policy_sha256`, hypothesis, logic/parameter diff, runtime, in-sample metrics, purged walk-forward OOS metrics, benchmark comparison, gate pass/fail breakdown, failure modes, and `PROMOTE` / `DISCARD` decision.
- `experiments/best_candidate/` — Snapshot of the single highest-scoring strategy that passed all validation and risk gates (`strategy.py`, `metrics.json`), plus a human-readable `MEMO.md` explaining what changed, why it beat the prior champion, and what weaknesses remain.
- `tests/` — Offline unit & regression tests verifying causality/no-lookahead, NaN/Inf rejection, transaction-cost drag, purge-gap separation, locked-holdout isolation, and policy immutability.

---

### 2. Core Recursive Loop (Execute in Bounded Cycles)
Run up to `MAX_CYCLES` (`config/risk_policy.json -> promotion.max_cycles_per_invocation`, default `3`) per invocation, stopping early after `stop_after_consecutive_discards` consecutive discards or any policy/integrity failure:

1. **Inspect State:** Verify `config/risk_policy.json` SHA-256, read the latest rows of `experiments/ledger.jsonl`, and review `experiments/best_candidate/MEMO.md`. Identify the single biggest weakness in the current champion (e.g., drawdown clustering, regime sensitivity, turnover drag, tail loss, or poor walk-forward OOS Sharpe).
2. **Formulate One Falsifiable Hypothesis:** State one clear, economically interpretable change (signal, regime filter, volatility scaling, rebalance cadence, or portfolio construction)—avoiding sprawling multi-variable rewrites in a single cycle.
3. **Implement & Test Deterministically:** Write a new candidate module in `src/strategies/` and run offline sanity & causality tests first (`pytest`: prefix-invariance lookahead test, zero NaN/Inf weights, gross leverage & single-asset weight caps, deterministic output).
4. **Evaluate Out-of-Sample (Purged Walk-Forward Only):** Run purged walk-forward OOS evaluation on the development window (`development_start`..`development_end`) via `src/engine/`. Never touch the locked holdout window during optimization. Compare against both the naive benchmark (`SPY` buy-and-hold) and the current `experiments/best_candidate/`.
5. **Audit & Decide (`PROMOTE` vs. `DISCARD`):**
   - **PROMOTE** only if the candidate improves the primary objective (`oos_sharpe`, net of costs) by at least `min_oos_sharpe_improvement` (`+0.05`, or establishes Cycle 0 baseline) **without** breaching any gate in `config/risk_policy.json` (max OOS drawdown, max annual turnover, min rebalance events, max IS-to-OOS Sharpe degradation, and parameter-perturbation stability).
   - **DISCARD** if metrics regress, fail to clear the minimum improvement margin, or breach any risk/sanity check—still append the full record to `experiments/ledger.jsonl` so future cycles do not repeat dead ends.
6. **Meta-Improvement (Harness Audit):** Every 3 cycles, inspect `src/engine/` and `tests/` for missing diagnostics (e.g., crisis-regime breakdown, cost-sensitivity sweep, tail-risk metrics) and verify any harness upgrade against offline regression tests before scoring new strategies.

---

### 3. Hard Guardrails & Anti-Overfitting Rules
- **No live execution or investment advice:** Historical simulation and research only.
- **No survivorship bias:** Default universe consists of liquid, long-lived multi-asset ETFs (`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `SHY`).
- **Locked holdout discipline:** `holdout_start` (`2024-01-02`) onward is sealed during the recursive loop and only opened on explicit human request for a one-time final audit.
- **Anti-curve-fitting discipline:** Track `trial_count`, require minimum OOS improvement margin, penalize IS→OOS Sharpe degradation, and reject parameter cliffs.
- **Honest reporting:** Every reported number must come from an executed command in this session; never cherry-pick windows or hide failed runs.

---

### 4. First Turn Action
1. Create the directory layout, immutable `config/risk_policy.json`, frozen dataset in `data/` with `manifest.json`, deterministic `src/engine/` harness, and offline `tests/`.
2. Run `pytest` and `ruff` to verify harness integrity.
3. Execute **Cycle 0 (Baseline)** (`v000_baseline.py`), log results to `experiments/ledger.jsonl` and `experiments/best_candidate/MEMO.md`, and report the verified baseline before launching optimization Cycles 1–N.
