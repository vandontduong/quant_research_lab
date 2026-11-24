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
### `cycle_007` — `v007_erc_risk_parity` (`DISCARD`)

- Verdict: DISCARD `v007_erc_risk_parity`: OOS Sharpe 0.818 (-0.221 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.221, Max DD worsened (-19.2% vs -16.2%), turnover increased (74% vs 39%).
- Drivers: Top contributor `QQQ` (+2.44%/yr), lagging `SHY` (+0.32%/yr). Diversification 3.90 eff. bets, avg cash 0.0%, SPY beta 0.39. Hardest stress regime `2022_rate_shock` (ret -12.5%, DD -15.8%). Fold Sharpe min/med/std=-0.01/1.02/0.81 (86% positive).
- Audit: IS→OOS gap -0.547; perturbation drop +0.073; DSR(n=8)=76.2%; 3x-cost Sharpe drag 0.016.
- Next: Signal materially underperformed champion (whipsaw or cash drag); avoid hard binary cutoffs and test continuous risk-budget adjustments instead.
### `cycle_008` — `v008_downside_vol_inv_weight` (`DISCARD`)

- Verdict: DISCARD `v008_downside_vol_inv_weight`: OOS Sharpe 1.040 (+0.002 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.002, Max DD held flat (-16.3% vs -16.2%), turnover increased (44% vs 39%).
- Drivers: Top contributor `QQQ` (+1.90%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.17 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.2%, DD -15.4%). Fold Sharpe min/med/std=0.25/0.84/0.71 (100% positive).
- Audit: IS→OOS gap -0.035; perturbation drop +0.031; DSR(n=9)=84.6%; 3x-cost Sharpe drag 0.015.
- Next: Positive but sub-margin gain (+0.002 < +0.050); test whether reducing turnover (44%/yr) or combining with a low-correlation defensive filter clears the bar.
### `cycle_009` — `v009_rate_regime_inv_vol` (`DISCARD`)

- Verdict: DISCARD `v009_rate_regime_inv_vol`: OOS Sharpe 1.042 (+0.003 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.003, Max DD held flat (-16.1% vs -16.2%), turnover kept flat (39% vs 39%).
- Drivers: Top contributor `QQQ` (+1.91%/yr), lagging `SHY` (+0.37%/yr). Diversification 4.14 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.0%, DD -15.3%). Fold Sharpe min/med/std=0.26/0.86/0.72 (100% positive).
- Audit: IS→OOS gap -0.042; perturbation drop +0.001; DSR(n=10)=69.1%; 3x-cost Sharpe drag 0.013.
- Next: Positive but sub-margin gain (+0.003 < +0.050); test whether reducing turnover (39%/yr) or combining with a low-correlation defensive filter clears the bar.
### `cycle_010` — `v010_dmn_sharpe_opt` (`DISCARD`)

- Verdict: DISCARD `v010_dmn_sharpe_opt`: OOS Sharpe 1.039 (+0.001 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.001, Max DD held flat (-16.2% vs -16.2%), turnover kept flat (40% vs 39%).
- Drivers: Top contributor `QQQ` (+1.90%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.15 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.2%, DD -15.4%). Fold Sharpe min/med/std=0.25/0.87/0.71 (100% positive).
- Audit: IS→OOS gap +0.040; perturbation drop +0.007; DSR(n=11)=48.1%; 3x-cost Sharpe drag 0.013.
- Next: Positive but sub-margin gain (+0.001 < +0.050); test whether reducing turnover (40%/yr) or combining with a low-correlation defensive filter clears the bar.
### `cycle_011` — `v011_momentum_transformer_cpd` (`DISCARD`)

- Verdict: DISCARD `v011_momentum_transformer_cpd`: OOS Sharpe 1.007 (-0.031 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.031, Max DD held flat (-16.3% vs -16.2%), turnover increased (59% vs 39%).
- Drivers: Top contributor `QQQ` (+1.99%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.28 eff. bets, avg cash 0.0%, SPY beta 0.29. Hardest stress regime `2022_rate_shock` (ret -12.4%, DD -15.6%). Fold Sharpe min/med/std=0.26/0.96/0.61 (100% positive).
- Audit: IS→OOS gap +0.075; perturbation drop +0.017; DSR(n=12)=23.9%; 3x-cost Sharpe drag 0.019.
- Next: Target `2022_rate_shock` drawdown (-15.6%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_012` — `v012_drl_vol_scaled_policy` (`DISCARD`)

- Verdict: DISCARD `v012_drl_vol_scaled_policy`: OOS Sharpe 1.037 (-0.002 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.002, Max DD held flat (-16.2% vs -16.2%), turnover kept flat (40% vs 39%).
- Drivers: Top contributor `QQQ` (+1.91%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.17 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.2%, DD -15.4%). Fold Sharpe min/med/std=0.26/0.86/0.71 (100% positive).
- Audit: IS→OOS gap -0.020; perturbation drop +0.002; DSR(n=13)=16.4%; 3x-cost Sharpe drag 0.013.
- Next: Target `2022_rate_shock` drawdown (-15.4%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_013` — `v013_sdf_macro_factor` (`DISCARD`)

- Verdict: DISCARD `v013_sdf_macro_factor`: OOS Sharpe 0.981 (-0.057 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.057, Max DD worsened (-16.9% vs -16.2%), turnover increased (77% vs 39%).
- Drivers: Top contributor `QQQ` (+1.96%/yr), lagging `SHY` (+0.40%/yr). Diversification 4.01 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -13.0%, DD -16.3%). Fold Sharpe min/med/std=0.22/0.79/0.64 (100% positive).
- Audit: IS→OOS gap -0.066; perturbation drop +0.081; DSR(n=14)=3.5%; 3x-cost Sharpe drag 0.025.
- Next: Signal materially underperformed champion (whipsaw or cash drag); avoid hard binary cutoffs and test continuous risk-budget adjustments instead.
### `cycle_014` — `v014_stat_arb_residual_mr` (`DISCARD`)

- Verdict: DISCARD `v014_stat_arb_residual_mr`: OOS Sharpe 1.020 (-0.019 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.019, Max DD worsened (-16.7% vs -16.2%), turnover increased (62% vs 39%).
- Drivers: Top contributor `QQQ` (+1.94%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.17 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.6%, DD -16.0%). Fold Sharpe min/med/std=0.27/0.82/0.71 (100% positive).
- Audit: IS→OOS gap +0.032; perturbation drop +0.003; DSR(n=15)=1.1%; 3x-cost Sharpe drag 0.020.
- Next: Target `2022_rate_shock` drawdown (-16.0%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_015` — `v015_eiie_portfolio_rl` (`DISCARD`)

- Verdict: DISCARD `v015_eiie_portfolio_rl`: OOS Sharpe 0.991 (-0.047 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.047, Max DD worsened (-18.6% vs -16.2%), turnover reduced (16% vs 39%).
- Drivers: Top contributor `QQQ` (+2.36%/yr), lagging `SHY` (+0.32%/yr). Diversification 4.83 eff. bets, avg cash 0.0%, SPY beta 0.36. Hardest stress regime `2022_rate_shock` (ret -14.6%, DD -17.7%). Fold Sharpe min/med/std=0.23/0.85/0.70 (100% positive).
- Audit: IS→OOS gap -0.298; perturbation drop +0.005; DSR(n=16)=0.1%; 3x-cost Sharpe drag 0.004.
- Next: Target `2022_rate_shock` drawdown (-17.7%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_016` — `v016_mm_deeplob_inventory` (`DISCARD`)

- Verdict: DISCARD `v016_mm_deeplob_inventory`: OOS Sharpe 1.024 (-0.015 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.015, Max DD held flat (-16.5% vs -16.2%), turnover kept flat (43% vs 39%).
- Drivers: Top contributor `QQQ` (+1.88%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.16 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.5%, DD -15.7%). Fold Sharpe min/med/std=0.23/0.92/0.70 (100% positive).
- Audit: IS→OOS gap +0.003; perturbation drop +0.012; DSR(n=17)=0.0%; 3x-cost Sharpe drag 0.014.
- Next: Target `2022_rate_shock` drawdown (-15.7%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_017` — `v017_news_underreaction_pead` (`DISCARD`)

- Verdict: DISCARD `v017_news_underreaction_pead`: OOS Sharpe 1.032 (-0.007 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe -0.007, Max DD held flat (-16.4% vs -16.2%), turnover increased (45% vs 39%).
- Drivers: Top contributor `QQQ` (+1.84%/yr), lagging `SHY` (+0.36%/yr). Diversification 4.14 eff. bets, avg cash 0.0%, SPY beta 0.27. Hardest stress regime `2022_rate_shock` (ret -12.4%, DD -15.6%). Fold Sharpe min/med/std=0.25/0.86/0.68 (100% positive).
- Audit: IS→OOS gap -0.027; perturbation drop +0.013; DSR(n=18)=0.0%; 3x-cost Sharpe drag 0.015.
- Next: Target `2022_rate_shock` drawdown (-15.6%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.
### `cycle_018` — `v018_trading_agents_debate` (`DISCARD`)

- Verdict: DISCARD `v018_trading_agents_debate`: OOS Sharpe 1.044 (+0.005 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.005, Max DD held flat (-16.2% vs -16.2%), turnover increased (44% vs 39%).
- Drivers: Top contributor `QQQ` (+1.94%/yr), lagging `SHY` (+0.37%/yr). Diversification 4.16 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.1%, DD -15.4%). Fold Sharpe min/med/std=0.27/0.87/0.71 (100% positive).
- Audit: IS→OOS gap -0.007; perturbation drop +0.003; DSR(n=19)=0.0%; 3x-cost Sharpe drag 0.015.
- Next: Positive but sub-margin gain (+0.005 < +0.050); test whether reducing turnover (44%/yr) or combining with a low-correlation defensive filter clears the bar.
### `cycle_019` — `v019_qlib_factor_ensemble` (`DISCARD`)

- Verdict: DISCARD `v019_qlib_factor_ensemble`: OOS Sharpe 1.042 (+0.003 vs `v001_inv_vol_monthly`); failed [beats_champion_margin].
- Evidence: vs `v001_inv_vol_monthly`: Sharpe +0.003, Max DD held flat (-16.1% vs -16.2%), turnover kept flat (40% vs 39%).
- Drivers: Top contributor `QQQ` (+1.88%/yr), lagging `SHY` (+0.37%/yr). Diversification 4.13 eff. bets, avg cash 0.0%, SPY beta 0.27. Hardest stress regime `2022_rate_shock` (ret -12.1%, DD -15.3%). Fold Sharpe min/med/std=0.25/0.84/0.72 (100% positive).
- Audit: IS→OOS gap -0.018; perturbation drop +0.000; DSR(n=20)=0.0%; 3x-cost Sharpe drag 0.013.
- Next: Positive but sub-margin gain (+0.003 < +0.050); test whether reducing turnover (40%/yr) or combining with a low-correlation defensive filter clears the bar.
