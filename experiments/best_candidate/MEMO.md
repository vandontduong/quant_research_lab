# Champion Memo — `v001_inv_vol_monthly` (cycle_001)

- **Promoted at:** `2026-11-23T18:07:15Z` (trial #2)
- **Policy SHA-256:** `3fef4ec1ef335cb2...` (verified immutable)
- **Dataset SHA-256:** `082a90981e57f2e9...` (development window only; holdout locked)
- **Evaluation Mode:** `fixed_param_causal_oos`
- **Scoring Convention:** Excess of `SHY` (residual cash earns `SHY`; Sharpe/Sortino/PSR/DSR on excess returns)
- **Prior champion: `v000_equal_weight_monthly` (OOS Sharpe `0.859`)**
- **Hypothesis:** Replace static equal-weight with trailing 63-day inverse-volatility weighting (with 4% annualized vol floor and 45% target single-asset cap, trimmed if drift > 49.5%) to equalize risk contributions and reduce drawdown clustering during equity/rate volatility spikes.
- **Parameters:** `{"vol_window": 63, "rebalance_every_days": 21, "min_vol_floor": 0.04, "max_weight": 0.45}`
- **Deflated Sharpe Ratio (DSR):** `0.9979` (trial #2, 3500 OOS observations)
- **Deflated Sharpe (N=20 historical trials):** `0.9458` (vs `N=2` at promotion: `0.9979`)
- **Perturbation Stability:** worst OOS Sharpe drop `0.039365`

> Note: Metrics refreshed on the Batch 2 methodology after fold-switch cost and DSR-moment fixes (policy `v1.1.0`, settled `2026-11-23` dataset) without appending a synthetic trial (`metrics_refreshed_utc=2026-11-24T08:01:38Z`). Historical `experiments/ledger.jsonl` and `experiments/results.tsv` rows are kept append-only as originally recorded under the prior raw-Sharpe convention and are not retroactively rewritten.

## Purged Walk-Forward Out-of-Sample vs. SPY Benchmark (Net of 5 bps/side)

| Metric | Candidate (OOS) | Candidate (IS) | SPY Buy & Hold (OOS) |
| --- | ---: | ---: | ---: |
| CAGR | 6.36% | 6.82% | 13.24% |
| Volatility | 6.12% | 6.71% | 17.33% |
| Sharpe (excess of SHY) | 0.915 | 0.419 | 0.743 |
| Raw Sharpe (total net) | 1.038 | 1.017 | 0.805 |
| Sortino (excess of SHY) | 1.278 | 0.600 | 1.035 |
| Deflated Sharpe (N=2) | 0.9979 | — | — |
| Max Drawdown | -16.15% | -11.59% | -33.72% |
| Calmar | 0.394 | 0.589 | 0.393 |
| Daily 95% CVaR | -0.90% | -0.90% | -2.67% |
| Annual Turnover | 38.7% | 48.7% | 0.0% |
| Rebalance Events | 167 | 60 | 0 |

## Walk-Forward Fold Breakdown

| Fold | Window | CAGR | Sharpe | Max DD |
| ---: | --- | ---: | ---: | ---: |
| 1 | `2010-02-04` to `2012-02-02` | 11.79% | 1.530 | -3.32% |
| 2 | `2012-02-03` to `2014-02-05` | 4.42% | 0.810 | -3.64% |
| 3 | `2014-02-06` to `2016-02-05` | 3.39% | 0.546 | -5.64% |
| 4 | `2016-02-08` to `2018-02-06` | 9.86% | 2.247 | -3.80% |
| 5 | `2018-02-07` to `2020-02-07` | 7.83% | 1.021 | -5.37% |
| 6 | `2020-02-10` to `2022-02-07` | 5.66% | 0.662 | -11.98% |
| 7 | `2022-02-08` to `2023-12-29` | 1.78% | 0.212 | -13.10% |

## Stress-Regime Breakdown (Development OOS)

| Regime | Window | Total Return | Max DD | Sharpe | Worst Day |
| --- | --- | ---: | ---: | ---: | ---: |
| `2011_euro_debt` | `2011-05-02` to `2011-10-31` | 3.04% | -3.16% | 0.593 | -1.05% |
| `2018_q4_selloff` | `2018-10-01` to `2018-12-31` | -2.94% | -4.40% | -2.222 | -1.38% |
| `2020_covid_crash` | `2020-02-19` to `2020-06-30` | 0.53% | -11.98% | -0.284 | -3.17% |
| `2022_rate_shock` | `2022-01-03` to `2022-12-30` | -12.16% | -15.37% | -1.047 | -2.15% |

## Transaction-Cost Sensitivity Sweep (OOS)

| Cost Scenario | bps / side | OOS CAGR | OOS Sharpe | OOS Max DD |
| --- | ---: | ---: | ---: | ---: |
| `1x_5bps` | 5.0 | 6.36% | 0.915 | -16.15% |
| `2x_10bps` | 10.0 | 6.32% | 0.908 | -16.18% |
| `3x_15bps` | 15.0 | 6.28% | 0.901 | -16.20% |

## Portfolio Attribution (Development OOS)

- **Exposure:** avg gross `100.0%`, cash `0.0%`, effective bets `4.14`, SPY beta `0.28` (corr `0.80`)

| Asset | Avg Weight | Annualized Contribution |
| --- | ---: | ---: |
| `SPY` | 13.0% | +1.63% |
| `QQQ` | 10.5% | +1.91% |
| `IWM` | 9.5% | +0.99% |
| `TLT` | 13.1% | +0.66% |
| `GLD` | 12.8% | +0.84% |
| `SHY` | 41.0% | +0.37% |

## Concise Reasoning Trace

- Verdict: PROMOTE `v001_inv_vol_monthly`: OOS Sharpe 0.915 (+0.056 vs `v000_equal_weight_monthly`), all gates passed.
- Evidence: vs `v000_equal_weight_monthly`: Sharpe +0.056, Max DD worsened (-16.2% vs 0.0%), turnover increased (39% vs 0%).
- Drivers: Top contributor `QQQ` (+1.91%/yr), lagging `SHY` (+0.37%/yr). Diversification 4.14 eff. bets, avg cash 0.0%, SPY beta 0.28. Hardest stress regime `2022_rate_shock` (ret -12.2%, DD -15.4%). Fold Sharpe min/med/std=0.21/0.81/0.68 (100% positive).
- Audit: IS→OOS gap -0.495; perturbation drop +0.039; DSR(n=2)=99.8%; 3x-cost Sharpe drag 0.013.
- Next: Target `2022_rate_shock` drawdown (-15.4%) using cross-asset correlation/rate-vol awareness without sacrificing bull-regime participation.

> Offline historical simulation only — not investment advice or live trading instructions.
