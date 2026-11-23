"""Purged walk-forward OOS evaluation, lookahead causality check, and perturbation stability check."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from engine.backtest import run_backtest, summarize_returns
from engine.policy import RiskPolicy

StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]


def assert_no_lookahead(
    strategy_fn: StrategyFn,
    prices: pd.DataFrame,
    params: dict[str, Any],
    *,
    probe_offsets: tuple[int, ...] = (260, 520, 900),
) -> None:
    """Prefix-invariance test: mutating future bars (> t) must not change weights on or before t."""
    full_w = strategy_fn(prices, params)
    n = len(prices)
    for idx in probe_offsets:
        if idx >= n - 5:
            continue
        truncated = prices.iloc[: idx + 1].copy()
        trunc_w = strategy_fn(truncated, params)
        lhs = full_w.iloc[: idx + 1].reindex(index=truncated.index, columns=prices.columns).fillna(0.0)
        rhs = trunc_w.reindex(index=truncated.index, columns=prices.columns).fillna(0.0)
        if not np.allclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float), atol=1e-10, rtol=0.0):
            raise AssertionError(
                f"LOOKAHEAD DETECTED at bar index {idx} ({prices.index[idx].strftime('%Y-%m-%d')}): "
                "weights changed when future prices were removed."
            )


def generate_walk_forward_folds(
    index: pd.DatetimeIndex,
    *,
    train_days: int,
    purge_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, int]]:
    """Generate expanding-train, purged walk-forward OOS folds strictly within the development window."""
    n = len(index)
    folds: list[dict[str, int]] = []
    train_end = train_days
    while train_end + purge_days + 20 < n:
        test_start = train_end + purge_days
        test_end = min(test_start + test_days, n)
        if test_end - test_start < 20:
            break
        folds.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "purge_start": train_end,
                "purge_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_end += step_days
    if not folds:
        raise ValueError("Insufficient development history to form at least one purged walk-forward fold.")
    return folds


STRESS_REGIMES: tuple[tuple[str, str, str], ...] = (
    ("2011_euro_debt", "2011-05-01", "2011-10-31"),
    ("2018_q4_selloff", "2018-10-01", "2018-12-31"),
    ("2020_covid_crash", "2020-02-19", "2020-06-30"),
    ("2022_rate_shock", "2022-01-03", "2022-12-30"),
)


def compute_regime_breakdown(
    oos_net: pd.Series,
    oos_to: pd.Series,
    *,
    regimes: tuple[tuple[str, str, str], ...] = STRESS_REGIMES,
) -> dict[str, dict[str, Any]]:
    """Evaluate total return, max drawdown, and Sharpe inside named historical OOS stress regimes."""
    out: dict[str, dict[str, Any]] = {}
    for name, start_s, end_s in regimes:
        mask = (oos_net.index >= pd.Timestamp(start_s)) & (oos_net.index <= pd.Timestamp(end_s))
        chunk_r = oos_net.loc[mask]
        chunk_to = oos_to.loc[mask]
        if len(chunk_r) < 5:
            continue
        m = summarize_returns(chunk_r, chunk_to, rebalance_events=int((chunk_to > 1e-4).sum()))
        out[name] = {
            "start_date": m["start_date"],
            "end_date": m["end_date"],
            "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "worst_day": m["worst_day"],
        }
    return out


def compute_cost_sensitivity(
    dev_prices: pd.DataFrame,
    weights_all: pd.DataFrame,
    folds: list[dict[str, int]],
    policy: RiskPolicy,
    *,
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    enforce_single_asset_cap: bool = True,
) -> dict[str, dict[str, Any]]:
    """Sweep transaction costs across 1x/2x/3x policy bps to quantify turnover fragility."""
    base_bps = float(policy.execution["commission_bps_per_side"]) + float(policy.execution["slippage_bps_per_side"])
    out: dict[str, dict[str, Any]] = {}
    for mult in multipliers:
        bt = run_backtest(
            dev_prices,
            weights_all,
            policy,
            enforce_single_asset_cap=enforce_single_asset_cap,
            cost_multiplier=mult,
        )
        net_all = bt["net_returns"]
        to_all = bt["turnover"]
        chunks_r = []
        chunks_to = []
        for f in folds:
            t_start_dt = dev_prices.index[f["test_start"]]
            t_end_dt = dev_prices.index[f["test_end"] - 1]
            chunks_r.append(net_all.loc[(net_all.index >= t_start_dt) & (net_all.index <= t_end_dt)])
            chunks_to.append(to_all.loc[(to_all.index >= t_start_dt) & (to_all.index <= t_end_dt)])
        oos_r = pd.concat(chunks_r).sort_index()
        oos_t = pd.concat(chunks_to).sort_index()
        m = summarize_returns(oos_r, oos_t, rebalance_events=int((oos_t > 1e-4).sum()))
        key = f"{mult:g}x_{base_bps * mult:g}bps"
        out[key] = {
            "cost_bps_per_side": round(base_bps * mult, 2),
            "oos_cagr": m["cagr"],
            "oos_sharpe": m["sharpe"],
            "oos_max_drawdown": m["max_drawdown"],
        }
    return out


def evaluate_walk_forward(
    strategy_fn: StrategyFn,
    dev_prices: pd.DataFrame,
    params: dict[str, Any],
    policy: RiskPolicy,
    *,
    enforce_single_asset_cap: bool = True,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run purged walk-forward evaluation strictly inside the development window (never touches holdout)."""
    holdout_start = pd.Timestamp(policy.splits["holdout_start"])
    if dev_prices.index[-1] >= holdout_start:
        raise ValueError(
            f"Holdout leakage blocked: dev_prices ends on {dev_prices.index[-1].strftime('%Y-%m-%d')}, "
            f"on/after locked holdout_start {holdout_start.strftime('%Y-%m-%d')}."
        )

    assert_no_lookahead(strategy_fn, dev_prices, params)

    splits = policy.splits
    folds = generate_walk_forward_folds(
        dev_prices.index,
        train_days=int(splits["walk_forward_train_days"]),
        purge_days=int(splits["walk_forward_purge_days"]),
        test_days=int(splits["walk_forward_test_days"]),
        step_days=int(splits["walk_forward_step_days"]),
    )

    # Compute causal weights once on dev_prices (prefix-invariance guarantees fold-by-fold equivalence).
    weights_all = strategy_fn(dev_prices, params)
    full_bt = run_backtest(
        dev_prices,
        weights_all,
        policy,
        enforce_single_asset_cap=enforce_single_asset_cap,
    )
    net_all = full_bt["net_returns"]
    to_all = full_bt["turnover"]

    # Initial in-sample segment = first train fold; OOS = stitched purged test folds.
    first_train_end_dt = dev_prices.index[folds[0]["train_end"] - 1]
    is_net = net_all.loc[net_all.index <= first_train_end_dt]
    is_to = to_all.loc[to_all.index <= first_train_end_dt]
    is_metrics = summarize_returns(is_net, is_to, rebalance_events=int((is_to > 1e-4).sum()))

    oos_net_chunks: list[pd.Series] = []
    oos_to_chunks: list[pd.Series] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_idx, f in enumerate(folds):
        t_start_dt = dev_prices.index[f["test_start"]]
        t_end_dt = dev_prices.index[f["test_end"] - 1]
        chunk_r = net_all.loc[(net_all.index >= t_start_dt) & (net_all.index <= t_end_dt)]
        chunk_to = to_all.loc[(to_all.index >= t_start_dt) & (to_all.index <= t_end_dt)]
        oos_net_chunks.append(chunk_r)
        oos_to_chunks.append(chunk_to)
        fold_m = summarize_returns(chunk_r, chunk_to, rebalance_events=int((chunk_to > 1e-4).sum()))
        fold_summaries.append({"fold": fold_idx + 1, **fold_m})

    oos_net = pd.concat(oos_net_chunks).sort_index()
    oos_to = pd.concat(oos_to_chunks).sort_index()
    oos_metrics = summarize_returns(oos_net, oos_to, rebalance_events=int((oos_to > 1e-4).sum()))

    result: dict[str, Any] = {
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "folds": fold_summaries,
        "fold_count": len(fold_summaries),
    }
    if include_diagnostics:
        result["regime_breakdown"] = compute_regime_breakdown(oos_net, oos_to)
        result["cost_sensitivity"] = compute_cost_sensitivity(
            dev_prices,
            weights_all,
            folds,
            policy,
            enforce_single_asset_cap=enforce_single_asset_cap,
        )
    return result


def evaluate_perturbation_stability(
    strategy_fn: StrategyFn,
    dev_prices: pd.DataFrame,
    params: dict[str, Any],
    neighbor_params_list: list[dict[str, Any]],
    policy: RiskPolicy,
    base_oos_sharpe: float,
) -> float | None:
    """Measure worst OOS Sharpe drop across neighboring parameter values (anti-curve-fitting check)."""
    if not neighbor_params_list:
        return None
    worst_drop = 0.0
    for alt_params in neighbor_params_list:
        res = evaluate_walk_forward(strategy_fn, dev_prices, alt_params, policy, include_diagnostics=False)
        alt_sharpe = float(res["out_of_sample"]["sharpe"])
        drop = base_oos_sharpe - alt_sharpe
        worst_drop = max(worst_drop, drop)
    return round(worst_drop, 6)
