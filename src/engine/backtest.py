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


def summarize_returns(
    net_returns: pd.Series,
    daily_turnover: pd.Series,
    *,
    rebalance_events: int,
) -> dict[str, Any]:
    """Compute annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, and turnover."""
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
    sharpe = (mean_daily / std_daily) * math.sqrt(TRADING_DAYS_PER_YEAR) if std_daily > 0 else 0.0

    downside = r[r < 0.0]
    downside_std = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 0.0
    sortino = (mean_daily / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0

    dd_series = wealth / wealth.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    worst_day = float(r.min())
    q05 = float(np.quantile(r.to_numpy(dtype=float), 0.05))
    tail = r[r <= q05]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else worst_day

    ann_turnover = float(daily_turnover.mean() * TRADING_DAYS_PER_YEAR)
    return {
        "start_date": r.index[0].strftime("%Y-%m-%d"),
        "end_date": r.index[-1].strftime("%Y-%m-%d"),
        "observations": len(r),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "annual_volatility": round(ann_vol, 6),
        "sharpe": round(sharpe, 6),
        "sortino": round(sortino, 6),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 6),
        "worst_day": round(worst_day, 6),
        "cvar_95_daily": round(cvar_95, 6),
        "annual_turnover": round(ann_turnover, 6),
        "rebalance_events": int(rebalance_events),
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
        gross_r = float(np.dot(w_target, ret_arr[i]))
        net_r = gross_r - (one_way_to * 2.0 * cost_rate)
        net_rets[i] = net_r

        # Drift weights by day's asset returns to start of next bar.
        denom = 1.0 + gross_r
        if denom > 0:
            prev_drifted = (w_target * (1.0 + ret_arr[i])) / denom
        else:
            prev_drifted = np.zeros(n_assets, dtype=float)

    # Skip initial `lag` warm-up bars where no position can legally be held yet.
    net_series = pd.Series(net_rets, index=prices.index, name="net_return").iloc[lag:]
    to_series = pd.Series(turnovers, index=prices.index, name="turnover").iloc[lag:]
    metrics = summarize_returns(net_series, to_series, rebalance_events=rebalance_events)
    return {
        "metrics": metrics,
        "net_returns": net_series,
        "turnover": to_series,
    }
