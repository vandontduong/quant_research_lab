"""Deterministic next-bar execution backtester with realistic turnover and cost accounting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from engine.policy import RiskPolicy

TRADING_DAYS_PER_YEAR = 252.0


def validate_weights(weights: pd.DataFrame, policy: RiskPolicy) -> None:
    """Validate target portfolio weights against policy ceilings and numerical sanity."""
    if weights.empty:
        raise ValueError("Strategy returned empty weights DataFrame.")
    vals = weights.to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Strategy weights contain NaN or Inf values.")

    exec_cfg = policy.execution
    allow_short = bool(exec_cfg.get("allow_short", False))
    max_gross = float(exec_cfg.get("max_gross_leverage", 1.0))
    max_single = float(exec_cfg.get("max_single_asset_weight", 0.5))

    if not allow_short and float(vals.min()) < -1e-9:
        raise ValueError(f"Short weight detected ({vals.min():.4f}) while allow_short=False.")
    gross = np.abs(vals).sum(axis=1)
    if float(gross.max()) > max_gross + 1e-6:
        raise ValueError(f"Gross leverage {gross.max():.4f} exceeds max_gross_leverage {max_gross:.4f}.")
    if float(np.abs(vals).max()) > max_single + 1e-6:
        raise ValueError(
            f"Single-asset weight {np.abs(vals).max():.4f} exceeds max_single_asset_weight {max_single:.4f}."
        )


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf_approx(p: float) -> float:
    """Acklam-free rational approximation for standard normal quantile (p in (0, 1))."""
    p_clamped = min(max(p, 1e-12), 1.0 - 1e-12)
    # Beasley-Springer-Moro approximation (accurate to ~1e-7 across (1e-6, 1-1e-6))
    a = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
    b = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
    c = (
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    )
    y = p_clamped - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
        return num / den
    r = p_clamped if y < 0 else (1.0 - p_clamped)
    r = math.log(-math.log(r))
    val = c[-1]
    for coeff in reversed(c[:-1]):
        val = val * r + coeff
    return -val if y < 0 else val


def compute_probabilistic_sharpe(
    sharpe_annual: float,
    *,
    observations: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark_sharpe_annual: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): P(true SR > benchmark SR)."""
    if observations < 3:
        return 0.5
    sr = sharpe_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    sr0 = benchmark_sharpe_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    denom_sq = 1.0 - skewness * sr + ((excess_kurtosis + 2.0) / 4.0) * (sr * sr)
    denom = math.sqrt(max(denom_sq, 1e-12))
    z = ((sr - sr0) * math.sqrt(observations - 1)) / denom
    return round(_norm_cdf(z), 6)


def compute_deflated_sharpe(
    sharpe_annual: float,
    *,
    observations: int,
    trial_count: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    sharpe_std_annual: float = 0.25,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado): PSR adjusted for multiple testing."""
    n_trials = max(int(trial_count), 1)
    if n_trials == 1:
        sr0_annual = 0.0
    else:
        euler_gamma = 0.5772156649015329
        z1 = _norm_ppf_approx(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf_approx(1.0 - 1.0 / (n_trials * math.e))
        sr0_annual = sharpe_std_annual * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    return compute_probabilistic_sharpe(
        sharpe_annual,
        observations=observations,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        benchmark_sharpe_annual=sr0_annual,
    )


def summarize_returns(
    net_returns: pd.Series,
    daily_turnover: pd.Series,
    *,
    rebalance_events: int,
    excess_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Compute annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, tail risk, and PSR.

    When `excess_returns` is provided, `sharpe`, `sortino`, and `psr_vs_zero` are computed on the
    excess-of-cash-proxy series while wealth-path metrics (CAGR, volatility, drawdown, tail risk,
    skew/kurt) and `raw_sharpe` remain on total `net_returns`.
    """
    r = net_returns.dropna().astype(float)
    if len(r) < 2:
        raise ValueError("At least 2 return observations are required to summarize performance.")

    wealth = (1.0 + r).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    days = (r.index[-1] - r.index[0]).days
    years = days / 365.25 if days > 0 else len(r) / TRADING_DAYS_PER_YEAR
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else -1.0

    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))
    ann_vol = std_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
    raw_sharpe = (mean_daily / std_daily) * math.sqrt(TRADING_DAYS_PER_YEAR) if std_daily > 0 else 0.0

    dd_series = wealth / wealth.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    worst_day = float(r.min())
    q05 = float(np.quantile(r.to_numpy(dtype=float), 0.05))
    tail = r[r <= q05]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else worst_day

    skew = float(r.skew()) if len(r) >= 3 and std_daily > 0 else 0.0
    ex_kurt = float(r.kurt()) if len(r) >= 4 and std_daily > 0 else 0.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(ex_kurt):
        ex_kurt = 0.0

    if excess_returns is not None:
        ex = excess_returns.reindex(r.index).dropna().astype(float)
        if len(ex) != len(r):
            raise ValueError("excess_returns must align with net_returns without missing values.")
        ex_mean = float(ex.mean())
        ex_std = float(ex.std(ddof=1))
        sharpe = (ex_mean / ex_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if ex_std > 0 else 0.0
        downside_std = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(dtype=float), 0.0) ** 2)))
        sortino = (ex_mean / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0
        ex_skew = float(ex.skew()) if len(ex) >= 3 and ex_std > 0 else 0.0
        ex_kurt_psr = float(ex.kurt()) if len(ex) >= 4 and ex_std > 0 else 0.0
        if not math.isfinite(ex_skew):
            ex_skew = 0.0
        if not math.isfinite(ex_kurt_psr):
            ex_kurt_psr = 0.0
        psr_vs_zero = compute_probabilistic_sharpe(
            sharpe,
            observations=len(ex),
            skewness=ex_skew,
            excess_kurtosis=ex_kurt_psr,
            benchmark_sharpe_annual=0.0,
        )
    else:
        sharpe = raw_sharpe
        downside_std = float(np.sqrt(np.mean(np.minimum(r.to_numpy(dtype=float), 0.0) ** 2)))
        sortino = (mean_daily / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0
        psr_vs_zero = compute_probabilistic_sharpe(
            sharpe,
            observations=len(r),
            skewness=skew,
            excess_kurtosis=ex_kurt,
            benchmark_sharpe_annual=0.0,
        )

    ann_turnover = float(daily_turnover.mean() * TRADING_DAYS_PER_YEAR)
    return {
        "start_date": r.index[0].strftime("%Y-%m-%d"),
        "end_date": r.index[-1].strftime("%Y-%m-%d"),
        "observations": len(r),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "annual_volatility": round(ann_vol, 6),
        "sharpe": round(sharpe, 6),
        "raw_sharpe": round(raw_sharpe, 6),
        "sortino": round(sortino, 6),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 6),
        "worst_day": round(worst_day, 6),
        "cvar_95_daily": round(cvar_95, 6),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(ex_kurt, 4),
        "psr_vs_zero": psr_vs_zero,
        "annual_turnover": round(ann_turnover, 6),
        "rebalance_events": int(rebalance_events),
    }


def compute_portfolio_attribution(
    prices: pd.DataFrame,
    effective_weights: pd.DataFrame,
    net_returns: pd.Series,
    *,
    benchmark_ticker: str = "SPY",
) -> dict[str, Any]:
    """Compute interpretable exposure, diversification (effective bets), asset contribution, and SPY beta."""
    idx = net_returns.index
    w = effective_weights.reindex(index=idx, columns=prices.columns).fillna(0.0).astype(float)
    asset_rets = prices.pct_change().fillna(0.0).reindex(index=idx, columns=prices.columns).fillna(0.0)

    avg_weights = {col: round(float(w[col].mean()), 4) for col in prices.columns}
    gross = w.abs().sum(axis=1)
    avg_gross = float(gross.mean())
    avg_cash = max(0.0, 1.0 - avg_gross)

    # Effective number of bets = 1 / HHI across invested weights (computed per bar, then averaged).
    w_arr = w.to_numpy(dtype=float)
    row_sums = np.abs(w_arr).sum(axis=1, keepdims=True)
    norm_w = np.divide(np.abs(w_arr), row_sums, out=np.zeros_like(w_arr), where=row_sums > 1e-12)
    hhi = (norm_w**2).sum(axis=1)
    eff_bets = np.divide(1.0, hhi, out=np.zeros_like(hhi), where=hhi > 1e-12)
    avg_eff_bets = float(eff_bets.mean()) if len(eff_bets) > 0 else 0.0

    # Annualized gross return contribution per asset: mean(w_i * r_i) * 252
    contrib_df = w * asset_rets
    contrib = {
        col: round(float(contrib_df[col].mean() * TRADING_DAYS_PER_YEAR), 5)
        for col in prices.columns
    }
    top_asset = max(contrib, key=contrib.get) if contrib else None
    bottom_asset = min(contrib, key=contrib.get) if contrib else None

    bench_corr = 0.0
    bench_beta = 0.0
    if benchmark_ticker in asset_rets.columns and len(net_returns) >= 5:
        b = asset_rets[benchmark_ticker].to_numpy(dtype=float)
        p = net_returns.to_numpy(dtype=float)
        b_var = float(np.var(b, ddof=1))
        p_std = float(np.std(p, ddof=1))
        b_std = float(np.std(b, ddof=1))
        if b_var > 1e-12 and p_std > 1e-12 and b_std > 1e-12:
            cov = float(np.cov(p, b, ddof=1)[0, 1])
            bench_beta = cov / b_var
            bench_corr = cov / (p_std * b_std)

    return {
        "avg_weights": avg_weights,
        "avg_gross_exposure": round(avg_gross, 4),
        "avg_cash_weight": round(avg_cash, 4),
        "avg_effective_bets": round(avg_eff_bets, 3),
        "annualized_contribution": contrib,
        "top_contributor": {"ticker": top_asset, "annualized_return": contrib[top_asset]}
        if top_asset
        else None,
        "bottom_contributor": {"ticker": bottom_asset, "annualized_return": contrib[bottom_asset]}
        if bottom_asset
        else None,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_correlation": round(bench_corr, 4),
        "benchmark_beta": round(bench_beta, 4),
    }


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    policy: RiskPolicy,
    *,
    enforce_single_asset_cap: bool = True,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Simulate a strategy with strict signal lag, weight-drift accounting, and transaction costs.

    Execution convention:
    - `target_weights.loc[t]` is decided using information available strictly through close `t`.
    - With `signal_lag_bars = 1`, those weights are applied to asset returns on day `t + 1`.
    - One-way turnover on day `t + 1` compares new target weights against prior drifted weights,
      charged `(commission_bps_per_side + slippage_bps_per_side) * cost_multiplier`.
    """
    aligned_w = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0).astype(float)
    if enforce_single_asset_cap:
        validate_weights(aligned_w, policy)

    exec_cfg = policy.execution
    lag = int(exec_cfg.get("signal_lag_bars", 1))
    if lag < 1:
        raise ValueError("signal_lag_bars must be >= 1 to prevent lookahead execution.")

    base_bps = float(exec_cfg["commission_bps_per_side"]) + float(exec_cfg["slippage_bps_per_side"])
    cost_rate = (base_bps * float(cost_multiplier)) / 10_000.0

    asset_returns = prices.pct_change().fillna(0.0)
    effective_w = aligned_w.shift(lag).fillna(0.0)

    cash_ticker = policy.cash_proxy_ticker
    use_cash_proxy = policy.sharpe_excess_of_cash_proxy and (cash_ticker is not None)
    if use_cash_proxy and cash_ticker not in prices.columns:
        raise ValueError(
            f"Cash proxy ticker '{cash_ticker}' is required when sharpe_excess_of_cash_proxy=True "
            "but is missing from prices."
        )
    cash_ret_arr = (
        asset_returns[cash_ticker].to_numpy(dtype=float) if use_cash_proxy and cash_ticker is not None else None
    )

    n_days, n_assets = asset_returns.shape
    ret_arr = asset_returns.to_numpy(dtype=float)
    w_arr = effective_w.to_numpy(dtype=float)

    net_rets = np.zeros(n_days, dtype=float)
    turnovers = np.zeros(n_days, dtype=float)
    prev_drifted = np.zeros(n_assets, dtype=float)
    rebalance_events = 0

    for i in range(n_days):
        w_target = w_arr[i]
        delta = np.abs(w_target - prev_drifted)
        one_way_to = float(delta.sum()) * 0.5
        if one_way_to > 1e-4:
            rebalance_events += 1
        turnovers[i] = one_way_to
        asset_gross_r = float(np.dot(w_target, ret_arr[i]))
        if cash_ret_arr is not None:
            gross_exposure = float(np.abs(w_target).sum())
            residual_cash = max(0.0, 1.0 - gross_exposure)
            gross_r = asset_gross_r + residual_cash * float(cash_ret_arr[i])
        else:
            gross_r = asset_gross_r
        net_r = gross_r - (one_way_to * 2.0 * cost_rate)
        net_rets[i] = net_r

        # Drift weights by total portfolio gross return (including cash yield) to start of next bar.
        denom = 1.0 + gross_r
        if denom > 0:
            prev_drifted = (w_target * (1.0 + ret_arr[i])) / denom
        else:
            prev_drifted = np.zeros(n_assets, dtype=float)

    # Skip initial `lag` warm-up bars where no position can legally be held yet.
    net_series = pd.Series(net_rets, index=prices.index, name="net_return").iloc[lag:]
    to_series = pd.Series(turnovers, index=prices.index, name="turnover").iloc[lag:]
    excess_series: pd.Series | None = None
    if use_cash_proxy and cash_ticker is not None:
        cash_series = asset_returns[cash_ticker].iloc[lag:]
        excess_series = (net_series - cash_series).rename("excess_return")
    eff_w_trimmed = effective_w.iloc[lag:]
    metrics = summarize_returns(
        net_series,
        to_series,
        rebalance_events=rebalance_events,
        excess_returns=excess_series,
    )
    attribution = compute_portfolio_attribution(
        prices,
        eff_w_trimmed,
        net_series,
        benchmark_ticker=policy.benchmark_ticker,
    )
    return {
        "metrics": metrics,
        "net_returns": net_series,
        "excess_returns": excess_series,
        "turnover": to_series,
        "effective_weights": eff_w_trimmed,
        "attribution": attribution,
    }
