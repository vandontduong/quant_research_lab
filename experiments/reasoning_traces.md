# Experiment Reasoning Traces
> Concise, evidence-backed traces auto-generated from `experiments/ledger.jsonl`.

### `cycle_000` — `v000_equal_weight_monthly` (`PROMOTE`)

- Verdict: PROMOTE `v000_equal_weight_monthly` (baseline): OOS Sharpe 0.946, Max DD -22.71%, Turnover 16.0%.
- Evidence: vs SPY OOS (0.805): spread +0.141, Max DD -22.7%.
- Drivers: Weakest WF fold #7 (`2022-02-08`..`2023-12-29`, Sharpe 0.21, DD -18.9%).
- Audit: IS→OOS gap -0.334; perturbation drop +0.000; complexity 1 params / 37 SLOC.
- Next: Target the weakest walk-forward fold while keeping parameter count and turnover low.
### `cycle_001` — `v001_inv_vol_monthly` (`PROMOTE`)

- Verdict: PROMOTE `v001_inv_vol_monthly`: OOS Sharpe 1.038 (+0.093 vs `v000_equal_weight_monthly`), all gates passed.
- Evidence: vs `v000_equal_weight_monthly`: Sharpe +0.093, Max DD reduced (-16.2% vs -22.7%), turnover increased (39% vs 16%).
- Drivers: Weakest WF fold #7 (`2022-02-08`..`2023-12-29`, Sharpe 0.26, DD -13.1%).
- Audit: IS→OOS gap -0.021; perturbation drop +0.028; complexity 4 params / 72 SLOC.
- Next: Target the weakest walk-forward fold while keeping parameter count and turnover low.
### `cycle_002` — `v002_inv_vol_trend_filter` (`DISCARD`)

- Verdict: DISCARD `v002_inv_vol_trend_filter`: OOS Sharpe 0.899 (-0.140 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.140, Max DD reduced (-8.6% vs -16.2%), turnover increased (97% vs 39%).
- Drivers: Weakest WF fold #3 (`2014-02-06`..`2016-02-05`, Sharpe 0.53, DD -4.4%).
- Audit: IS→OOS gap +0.259; perturbation drop +0.017; complexity 6 params / 77 SLOC.
- Next: Signal materially underperformed champion (whipsaw or cash drag); avoid hard binary cutoffs and test continuous risk-budget adjustments instead.
### `cycle_003` — `v003_inv_vol_top_momentum` (`DISCARD`)

- Verdict: DISCARD `v003_inv_vol_top_momentum`: OOS Sharpe 0.689 (-0.350 vs `v001_inv_vol_monthly`); failed [max_oos_drawdown, beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.350, Max DD worsened (-27.4% vs -16.2%), turnover increased (261% vs 39%).
- Drivers: Weakest WF fold #7 (`2022-02-08`..`2023-12-29`, Sharpe -0.06, DD -18.6%).
- Audit: IS→OOS gap -0.086; perturbation drop +0.003; complexity 6 params / 68 SLOC.
- Next: Drawdown (-27.4%) breached policy ceiling; add volatility scaling or defensive-sleeve floor rather than concentrating into high-beta momentum.
### `cycle_004` — `v004_corr_aware_risk_parity` (`DISCARD`)

- Verdict: DISCARD `v004_corr_aware_risk_parity`: OOS Sharpe 1.045 (+0.006 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.006, Max DD held flat (-16.1% vs -16.2%), turnover kept flat (39% vs 39%).
- Drivers: Hardest stress regime `2022_rate_shock` (ret -12.0%, DD -15.3%).
- Audit: IS→OOS gap +0.020; perturbation drop +0.022; 3x-cost Sharpe drag 0.014; complexity 5 params / 64 SLOC.
- Next: Positive but sub-margin gain (+0.006 < +0.050); test whether reducing turnover (39%/yr) or combining with a low-correlation defensive filter clears the bar.
### `cycle_005` — `v005_soft_trend_inv_vol` (`DISCARD`)

- Verdict: DISCARD `v005_soft_trend_inv_vol`: OOS Sharpe 0.951 (-0.088 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.088, Max DD held flat (-16.6% vs -16.2%), turnover increased (90% vs 39%).
- Drivers: Hardest stress regime `2022_rate_shock` (ret -13.1%, DD -16.0%).
- Audit: IS→OOS gap -0.025; perturbation drop +0.031; 3x-cost Sharpe drag 0.026; complexity 7 params / 73 SLOC.
- Next: Signal materially underperformed champion (whipsaw or cash drag); avoid hard binary cutoffs and test continuous risk-budget adjustments instead.
### `cycle_006` — `v006_vol_targeted_inv_vol` (`DISCARD`)

- Verdict: DISCARD `v006_vol_targeted_inv_vol`: OOS Sharpe 1.055 (+0.017 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.017, Max DD worsened (-18.2% vs -16.2%), turnover increased (149% vs 39%).
- Drivers: Hardest stress regime `2022_rate_shock` (ret -14.3%, DD -17.1%).
- Audit: IS→OOS gap -0.182; perturbation drop +0.033; 3x-cost Sharpe drag 0.037; complexity 7 params / 79 SLOC.
- Next: Positive but sub-margin gain (+0.017 < +0.050); test whether reducing turnover (149%/yr) or combining with a low-correlation defensive filter clears the bar.
