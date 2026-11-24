# Champion Memo — `v001_inv_vol_monthly` (cycle_001)

- **Promoted at:** `2026-11-23T18:07:15Z` (trial #2)
- **Policy SHA-256:** `b8b1dd5fc55778a6...` (verified immutable)
- **Dataset SHA-256:** `7cd3f749d2fbb0b7...` (development window only; holdout locked)
- **Prior champion: `v000_equal_weight_monthly` (OOS Sharpe `0.946`)**
- **Hypothesis:** Replace static equal-weight with trailing 63-day inverse-volatility weighting (with 4% annualized vol floor and 45% target single-asset cap, trimmed if drift > 49.5%) to equalize risk contributions and reduce drawdown clustering during equity/rate volatility spikes.
- **Parameters:** `{"vol_window": 63, "rebalance_every_days": 21, "min_vol_floor": 0.04, "max_weight": 0.45}`
- **Deflated Sharpe Ratio (DSR):** `0.9996` at promotion (`trial_count = 2`, 3,500 OOS observations); `0.9804` evaluated at `N = 20` current trials.
- **Perturbation Stability:** worst OOS Sharpe drop `0.039` (`0.038646`) across 6 `NEIGHBOR_PARAMS` variants (policy ceiling `0.200`).

> Note: Historical `experiments/ledger.jsonl` Sortino and DSR fields were computed with prior functions and are not retroactively rewritten.

## Purged Walk-Forward Out-of-Sample vs. SPY Benchmark (Net of 5 bps/side)

| Metric | Candidate (OOS) | Candidate (IS) | SPY Buy & Hold (OOS) |
| --- | ---: | ---: | ---: |
| CAGR | 6.36% | 6.82% | 13.24% |
| Volatility | 6.12% | 6.71% | 17.33% |
| Sharpe (net) | 1.038 | 1.017 | 0.805 |
| Sortino | 1.471 | 1.509 | 1.127 |
| Deflated Sharpe (N=2) | 0.9996 | — | — |
| Perturbation Sharpe Drop | 0.039 | — | — |
| Max Drawdown | -16.15% | -11.59% | -33.72% |
| Calmar | 0.394 | 0.589 | 0.393 |
| Annual Turnover | 38.7% | 48.7% | 0.0% |
| Rebalance Events | 167 | 60 | 0 |

## Walk-Forward Fold Breakdown

| Fold | Window | CAGR | Sharpe | Max DD |
| ---: | --- | ---: | ---: | ---: |
| 1 | `2010-02-04` to `2012-02-02` | 11.79% | 1.821 | -3.32% |
| 2 | `2012-02-03` to `2014-02-05` | 4.42% | 0.860 | -3.64% |
| 3 | `2014-02-06` to `2016-02-05` | 3.39% | 0.685 | -5.64% |
| 4 | `2016-02-08` to `2018-02-06` | 9.86% | 2.205 | -3.80% |
| 5 | `2018-02-07` to `2020-02-07` | 7.83% | 1.598 | -5.37% |
| 6 | `2020-02-10` to `2022-02-07` | 5.66% | 0.723 | -11.98% |
| 7 | `2022-02-08` to `2023-12-29` | 1.78% | 0.256 | -13.10% |

## Observed Weaknesses to Target in Next Cycle
- Inspect worst walk-forward fold drawdown and Sharpe dispersion across regimes (especially 2022 rate shock vs. 2010–2021 low-rate regime).

> Offline historical simulation only — not investment advice or live trading instructions.
