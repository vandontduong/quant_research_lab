# Champion Memo — `v001_inv_vol_monthly` (cycle_001)

- **Promoted at:** `2026-11-23T18:07:15Z` (trial #2)
- **Policy SHA-256:** `b8b1dd5fc55778a6...` (verified immutable)
- **Dataset SHA-256:** `7cd3f749d2fbb0b7...` (development window only; holdout locked)
- **Prior champion: `v000_equal_weight_monthly` (OOS Sharpe `0.946`)**
- **Hypothesis:** Replace static equal-weight with trailing 63-day inverse-volatility weighting (with 4% annualized vol floor and 45% target single-asset cap, trimmed if drift > 49.5%) to equalize risk contributions and reduce drawdown clustering during equity/rate volatility spikes.
- **Parameters:** `{"vol_window": 63, "rebalance_every_days": 21, "min_vol_floor": 0.04, "max_weight": 0.45}`

## Purged Walk-Forward Out-of-Sample vs. SPY Benchmark (Net of 5 bps/side)

| Metric | Candidate (OOS) | Candidate (IS) | SPY Buy & Hold (OOS) |
| --- | ---: | ---: | ---: |
| CAGR | 6.36% | 6.82% | 13.24% |
| Volatility | 6.12% | 6.71% | 17.33% |
| Sharpe (net) | 1.038 | 1.017 | 0.805 |
| Sortino | 0.979 | 1.017 | 0.753 |
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
- Static equal-weight allocation does not adapt to volatility or trend regimes.

> Offline historical simulation only — not investment advice or live trading instructions.

## One-Time Sealed Holdout Audit (`2024-01-02` to `2026-11-23`, 727 bars)

- **Audited at:** `2026-11-23T19:18:54Z` (full audit saved in `experiments/holdout_audit.json`)
- **Result:** Champion `v001_inv_vol_monthly` generalized cleanly out-of-sample with **zero Sharpe degradation** (`1.038` WF-OOS → `1.209` sealed holdout), beating both `v000` baseline (`1.058`) and `SPY` (`1.052`) on net Sharpe, Sortino, Calmar, and max drawdown.

| Metric (Holdout, 5 bps/side) | Champion `v001` | Baseline `v000` | `SPY` Buy & Hold |
| --- | ---: | ---: | ---: |
| CAGR | 8.66% | 12.03% | 16.90% |
| Volatility | 7.09% | 11.37% | 16.11% |
| Net Sharpe (5 bps) | **1.209** | 1.058 | 1.052 |
| Net Sharpe (15 bps / 3x) | **1.198** | 1.055 | 1.052 |
| Sortino | **1.189** | 1.043 | 1.033 |
| Max Drawdown | **-6.27%** | -10.70% | -18.76% |
| Calmar | **1.380** | 1.124 | 0.901 |
| Worst Day | **-1.76%** | -2.86% | -5.85% |
| Daily 95% CVaR | **-1.00%** | -1.62% | -2.32% |
| Annual Turnover | 38.2% | 17.0% | 0.0% |
