"""Purged walk-forward OOS evaluation, lookahead causality check, and perturbation stability check."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from engine.backtest import compute_portfolio_attribution, run_backtest, summarize_returns
from engine.policy import RiskPolicy

StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]
FitFn = Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]]


def _to_json_safe(value: Any) -> Any:
    """Convert fitted parameter objects into JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, pd.Series):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def assert_no_lookahead(
    strategy_fn: StrategyFn,
    prices: pd.DataFrame,
    params: dict[str, Any],
    *,
    probe_offsets: tuple[int, ...] = (260, 520, 900),
) -> None:
    """Verify determinism, input immutability, prefix-truncation invariance, and future-shock invariance."""
    prices_snapshot = prices.copy(deep=True)
    params_snapshot = copy.deepcopy(params)

    full_w = strategy_fn(prices, params)
    if not prices.equals(prices_snapshot):
        raise AssertionError("INPUT MUTATION DETECTED: strategy mutated the `prices` DataFrame in-place.")
    if params != params_snapshot:
        raise AssertionError("INPUT MUTATION DETECTED: strategy mutated the `params` dict in-place.")

    # Determinism check: a second call on identical inputs must return identical weights.
    second_w = strategy_fn(prices, params)
    lhs_det = full_w.reindex(index=prices.index, columns=prices.columns).fillna(0.0).to_numpy(dtype=float)
    rhs_det = second_w.reindex(index=prices.index, columns=prices.columns).fillna(0.0).to_numpy(dtype=float)
    if not np.allclose(lhs_det, rhs_det, atol=1e-12, rtol=0.0):
        raise AssertionError("NONDETERMINISM DETECTED: repeated strategy calls produced different weights.")

    n = len(prices)
    for idx in probe_offsets:
        if idx >= n - 5:
            continue

        # 1) Prefix-truncation probe: removing future bars (> idx) must not alter weights <= idx.
        truncated = prices.iloc[: idx + 1].copy()
        trunc_w = strategy_fn(truncated, params)
        lhs = full_w.iloc[: idx + 1].reindex(index=truncated.index, columns=prices.columns).fillna(0.0)
        rhs = trunc_w.reindex(index=truncated.index, columns=prices.columns).fillna(0.0)
        if not np.allclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float), atol=1e-10, rtol=0.0):
            raise AssertionError(
                f"LOOKAHEAD DETECTED at bar index {idx} ({prices.index[idx].strftime('%Y-%m-%d')}): "
                "weights changed when future prices were removed."
            )

        # 2) Fixed-length future-shock probe: shocking prices strictly after idx must not alter weights <= idx.
        shocked = prices.copy()
        shocked.iloc[idx + 1 :] = shocked.iloc[idx + 1 :] * 1.5
        shocked_w = strategy_fn(shocked, params)
        rhs_shock = (
            shocked_w.iloc[: idx + 1].reindex(index=truncated.index, columns=prices.columns).fillna(0.0)
        )
        if not np.allclose(lhs.to_numpy(dtype=float), rhs_shock.to_numpy(dtype=float), atol=1e-10, rtol=0.0):
            raise AssertionError(
                f"LOOKAHEAD DETECTED (future-shock) at bar index {idx} ({prices.index[idx].strftime('%Y-%m-%d')}): "
                "weights changed when future prices were shocked."
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
    oos_excess: pd.Series | None = None,
    regimes: tuple[tuple[str, str, str], ...] = STRESS_REGIMES,
) -> dict[str, dict[str, Any]]:
    """Evaluate total return, max drawdown, and Sharpe inside named historical OOS stress regimes."""
    out: dict[str, dict[str, Any]] = {}
    for name, start_s, end_s in regimes:
        mask = (oos_net.index >= pd.Timestamp(start_s)) & (oos_net.index <= pd.Timestamp(end_s))
        chunk_r = oos_net.loc[mask]
        chunk_to = oos_to.loc[mask]
        chunk_ex = oos_excess.loc[mask] if oos_excess is not None else None
        if len(chunk_r) < 5:
            continue
        m = summarize_returns(
            chunk_r,
            chunk_to,
            rebalance_events=int((chunk_to > 1e-4).sum()),
            excess_returns=chunk_ex,
        )
        out[name] = {
            "start_date": m["start_date"],
            "end_date": m["end_date"],
            "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "raw_sharpe": m["raw_sharpe"],
            "worst_day": m["worst_day"],
        }
    return out


def compute_cost_sensitivity(
    dev_prices: pd.DataFrame,
    weights_all: pd.DataFrame | None,
    folds: list[dict[str, int]],
    policy: RiskPolicy,
    *,
    per_fold_weights: list[pd.DataFrame] | None = None,
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    enforce_single_asset_cap: bool = True,
) -> dict[str, dict[str, Any]]:
    """Sweep transaction costs across 1x/2x/3x policy bps to quantify turnover fragility."""
    if per_fold_weights is None and weights_all is None:
        raise ValueError("Either weights_all or per_fold_weights must be provided.")
    base_bps = float(policy.execution["commission_bps_per_side"]) + float(policy.execution["slippage_bps_per_side"])
    out: dict[str, dict[str, Any]] = {}
    for mult in multipliers:
        chunks_r: list[pd.Series] = []
        chunks_to: list[pd.Series] = []
        chunks_ex: list[pd.Series] = []

        if per_fold_weights is not None:
            for f, fold_w in zip(folds, per_fold_weights, strict=True):
                prefix_prices = dev_prices.iloc[: f["test_end"]]
                bt = run_backtest(
                    prefix_prices,
                    fold_w,
                    policy,
                    enforce_single_asset_cap=enforce_single_asset_cap,
                    cost_multiplier=mult,
                )
                net_f = bt["net_returns"]
                to_f = bt["turnover"]
                ex_f = bt["excess_returns"]
                t_start_dt = dev_prices.index[f["test_start"]]
                t_end_dt = dev_prices.index[f["test_end"] - 1]
                mask = (net_f.index >= t_start_dt) & (net_f.index <= t_end_dt)
                chunks_r.append(net_f.loc[mask])
                chunks_to.append(to_f.loc[mask])
                if ex_f is not None:
                    chunks_ex.append(ex_f.loc[mask])
        else:
            assert weights_all is not None
            bt = run_backtest(
                dev_prices,
                weights_all,
                policy,
                enforce_single_asset_cap=enforce_single_asset_cap,
                cost_multiplier=mult,
            )
            net_all = bt["net_returns"]
            to_all = bt["turnover"]
            ex_all = bt["excess_returns"]
            for f in folds:
                t_start_dt = dev_prices.index[f["test_start"]]
                t_end_dt = dev_prices.index[f["test_end"] - 1]
                mask = (net_all.index >= t_start_dt) & (net_all.index <= t_end_dt)
                chunks_r.append(net_all.loc[mask])
                chunks_to.append(to_all.loc[mask])
                if ex_all is not None:
                    chunks_ex.append(ex_all.loc[mask])

        oos_r = pd.concat(chunks_r).sort_index()
        oos_t = pd.concat(chunks_to).sort_index()
        oos_ex = pd.concat(chunks_ex).sort_index() if chunks_ex else None
        m = summarize_returns(
            oos_r,
            oos_t,
            rebalance_events=int((oos_t > 1e-4).sum()),
            excess_returns=oos_ex,
        )
        key = f"{mult:g}x_{base_bps * mult:g}bps"
        out[key] = {
            "cost_bps_per_side": round(base_bps * mult, 2),
            "oos_cagr": m["cagr"],
            "oos_sharpe": m["sharpe"],
            "oos_raw_sharpe": m["raw_sharpe"],
            "oos_max_drawdown": m["max_drawdown"],
        }
    return out


def _compute_fold_consistency(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    fold_sharpes = [float(f["sharpe"]) for f in fold_summaries]
    return {
        "min_fold_sharpe": round(min(fold_sharpes), 4),
        "max_fold_sharpe": round(max(fold_sharpes), 4),
        "median_fold_sharpe": round(float(np.median(fold_sharpes)), 4),
        "std_fold_sharpe": round(float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0, 4),
        "positive_fold_ratio": round(sum(1 for s in fold_sharpes if s > 0.0) / len(fold_sharpes), 4),
        "worst_fold": min(fold_summaries, key=lambda f: float(f["sharpe"]))["fold"],
        "best_fold": max(fold_summaries, key=lambda f: float(f["sharpe"]))["fold"],
    }


def evaluate_walk_forward(
    strategy_fn: StrategyFn,
    dev_prices: pd.DataFrame,
    params: dict[str, Any],
    policy: RiskPolicy,
    *,
    fit_fn: FitFn | None = None,
    enforce_single_asset_cap: bool = True,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run purged walk-forward evaluation strictly inside the development window (never touches holdout).

    When `fit_fn` is None (`evaluation_mode="fixed_param_causal_oos"`), `strategy_fn` is called once
    across `dev_prices` and sliced into purged folds (valid for fixed-parameter causal strategies).
    When `fit_fn` is provided (`evaluation_mode="per_fold_purged_wf"`), `fit_fn(train_prices, params)`
    is invoked per fold strictly on `[train_start:train_end]` before causal weight generation through
    `test_end`.
    """
    holdout_start = pd.Timestamp(policy.splits["holdout_start"])
    if dev_prices.index[-1] >= holdout_start:
        raise ValueError(
            f"Holdout leakage blocked: dev_prices ends on {dev_prices.index[-1].strftime('%Y-%m-%d')}, "
            f"on/after locked holdout_start {holdout_start.strftime('%Y-%m-%d')}."
        )

    splits = policy.splits
    purge_days = int(splits["walk_forward_purge_days"])
    folds = generate_walk_forward_folds(
        dev_prices.index,
        train_days=int(splits["walk_forward_train_days"]),
        purge_days=purge_days,
        test_days=int(splits["walk_forward_test_days"]),
        step_days=int(splits["walk_forward_step_days"]),
    )

    if fit_fn is None:
        assert_no_lookahead(strategy_fn, dev_prices, params)

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
        ex_all = full_bt["excess_returns"]

        # Initial in-sample segment = first train fold; OOS = stitched purged test folds.
        first_train_end_dt = dev_prices.index[folds[0]["train_end"] - 1]
        is_mask = net_all.index <= first_train_end_dt
        is_net = net_all.loc[is_mask]
        is_to = to_all.loc[is_mask]
        is_ex = ex_all.loc[is_mask] if ex_all is not None else None
        is_metrics = summarize_returns(
            is_net,
            is_to,
            rebalance_events=int((is_to > 1e-4).sum()),
            excess_returns=is_ex,
        )

        oos_net_chunks: list[pd.Series] = []
        oos_to_chunks: list[pd.Series] = []
        oos_ex_chunks: list[pd.Series] = []
        fold_summaries: list[dict[str, Any]] = []

        for fold_idx, f in enumerate(folds):
            t_start_dt = dev_prices.index[f["test_start"]]
            t_end_dt = dev_prices.index[f["test_end"] - 1]
            mask = (net_all.index >= t_start_dt) & (net_all.index <= t_end_dt)
            chunk_r = net_all.loc[mask]
            chunk_to = to_all.loc[mask]
            chunk_ex = ex_all.loc[mask] if ex_all is not None else None
            oos_net_chunks.append(chunk_r)
            oos_to_chunks.append(chunk_to)
            if chunk_ex is not None:
                oos_ex_chunks.append(chunk_ex)
            fold_m = summarize_returns(
                chunk_r,
                chunk_to,
                rebalance_events=int((chunk_to > 1e-4).sum()),
                excess_returns=chunk_ex,
            )
            fold_summaries.append({"fold": fold_idx + 1, **fold_m})

        oos_net = pd.concat(oos_net_chunks).sort_index()
        oos_to = pd.concat(oos_to_chunks).sort_index()
        oos_ex = pd.concat(oos_ex_chunks).sort_index() if oos_ex_chunks else None
        oos_metrics = summarize_returns(
            oos_net,
            oos_to,
            rebalance_events=int((oos_to > 1e-4).sum()),
            excess_returns=oos_ex,
        )

        result: dict[str, Any] = {
            "evaluation_mode": "fixed_param_causal_oos",
            "in_sample": is_metrics,
            "out_of_sample": oos_metrics,
            "folds": fold_summaries,
            "fold_count": len(fold_summaries),
            "fold_consistency": _compute_fold_consistency(fold_summaries),
        }
        if include_diagnostics:
            result["attribution"] = compute_portfolio_attribution(
                dev_prices,
                full_bt["effective_weights"],
                oos_net,
                benchmark_ticker=policy.benchmark_ticker,
            )
            result["regime_breakdown"] = compute_regime_breakdown(oos_net, oos_to, oos_excess=oos_ex)
            result["cost_sensitivity"] = compute_cost_sensitivity(
                dev_prices,
                weights_all,
                folds,
                policy,
                enforce_single_asset_cap=enforce_single_asset_cap,
            )
        return result

    # Per-fold refit path: fit_fn sees ONLY [train_start:train_end] per fold with purge gap enforced.
    oos_net_chunks = []
    oos_to_chunks = []
    oos_ex_chunks = []
    oos_eff_w_chunks: list[pd.DataFrame] = []
    per_fold_weights: list[pd.DataFrame] = []
    fold_summaries = []
    fitted_params_by_fold: list[dict[str, Any]] = []
    is_metrics: dict[str, Any] | None = None

    for fold_idx, f in enumerate(folds):
        tr_start = int(f["train_start"])
        tr_end = int(f["train_end"])
        te_start = int(f["test_start"])
        te_end = int(f["test_end"])

        if not (0 <= tr_start < tr_end and tr_end + purge_days <= te_start < te_end <= len(dev_prices)):
            raise ValueError(
                f"Invalid fold {fold_idx + 1} purge geometry: "
                f"train=[{tr_start}:{tr_end}], purge_days={purge_days}, test=[{te_start}:{te_end}]."
            )

        train_prices = dev_prices.iloc[tr_start:tr_end].copy(deep=True)
        fitted_params = fit_fn(train_prices, copy.deepcopy(params))
        if not isinstance(fitted_params, dict):
            raise TypeError(f"fit_fn must return a dict of fitted parameters, got {type(fitted_params).__name__}.")

        if fold_idx == 0:
            # Verify strategy_fn causality using representative fitted params without letting lookahead
            # probes re-invoke fit_fn past train_end.
            assert_no_lookahead(strategy_fn, dev_prices, copy.deepcopy(fitted_params))

        prefix_prices = dev_prices.iloc[:te_end]
        fold_weights = strategy_fn(prefix_prices, copy.deepcopy(fitted_params))
        per_fold_weights.append(fold_weights)

        fold_bt = run_backtest(
            prefix_prices,
            fold_weights,
            policy,
            enforce_single_asset_cap=enforce_single_asset_cap,
        )
        net_f = fold_bt["net_returns"]
        to_f = fold_bt["turnover"]
        ex_f = fold_bt["excess_returns"]
        eff_w_f = fold_bt["effective_weights"]

        tr_start_dt = dev_prices.index[tr_start]
        tr_end_dt = dev_prices.index[tr_end - 1]
        tr_mask = (net_f.index >= tr_start_dt) & (net_f.index <= tr_end_dt)
        fold_is_m = summarize_returns(
            net_f.loc[tr_mask],
            to_f.loc[tr_mask],
            rebalance_events=int((to_f.loc[tr_mask] > 1e-4).sum()),
            excess_returns=ex_f.loc[tr_mask] if ex_f is not None else None,
        )
        if fold_idx == 0:
            is_metrics = fold_is_m

        te_start_dt = dev_prices.index[te_start]
        te_end_dt = dev_prices.index[te_end - 1]
        te_mask = (net_f.index >= te_start_dt) & (net_f.index <= te_end_dt)
        chunk_r = net_f.loc[te_mask]
        chunk_to = to_f.loc[te_mask]
        chunk_ex = ex_f.loc[te_mask] if ex_f is not None else None
        chunk_w = eff_w_f.loc[(eff_w_f.index >= te_start_dt) & (eff_w_f.index <= te_end_dt)]

        oos_net_chunks.append(chunk_r)
        oos_to_chunks.append(chunk_to)
        if chunk_ex is not None:
            oos_ex_chunks.append(chunk_ex)
        oos_eff_w_chunks.append(chunk_w)

        fold_m = summarize_returns(
            chunk_r,
            chunk_to,
            rebalance_events=int((chunk_to > 1e-4).sum()),
            excess_returns=chunk_ex,
        )
        fitted_summary = _to_json_safe(fitted_params)
        fold_summaries.append(
            {
                "fold": fold_idx + 1,
                **fold_m,
                "in_sample_sharpe": fold_is_m["sharpe"],
                "fitted_params": fitted_summary,
            }
        )
        fitted_params_by_fold.append(
            {
                "fold": fold_idx + 1,
                "train_start": tr_start,
                "train_end": tr_end,
                "test_start": te_start,
                "test_end": te_end,
                "train_start_date": tr_start_dt.strftime("%Y-%m-%d"),
                "train_end_date": tr_end_dt.strftime("%Y-%m-%d"),
                "test_start_date": te_start_dt.strftime("%Y-%m-%d"),
                "test_end_date": te_end_dt.strftime("%Y-%m-%d"),
                "fitted_params": fitted_summary,
            }
        )

    assert is_metrics is not None
    oos_net = pd.concat(oos_net_chunks).sort_index()
    oos_to = pd.concat(oos_to_chunks).sort_index()
    oos_ex = pd.concat(oos_ex_chunks).sort_index() if oos_ex_chunks else None
    oos_eff_w = pd.concat(oos_eff_w_chunks).sort_index()
    oos_metrics = summarize_returns(
        oos_net,
        oos_to,
        rebalance_events=int((oos_to > 1e-4).sum()),
        excess_returns=oos_ex,
    )

    result = {
        "evaluation_mode": "per_fold_purged_wf",
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "folds": fold_summaries,
        "fold_count": len(fold_summaries),
        "fold_consistency": _compute_fold_consistency(fold_summaries),
        "fitted_params_by_fold": fitted_params_by_fold,
    }
    if include_diagnostics:
        result["attribution"] = compute_portfolio_attribution(
            dev_prices,
            oos_eff_w,
            oos_net,
            benchmark_ticker=policy.benchmark_ticker,
        )
        result["regime_breakdown"] = compute_regime_breakdown(oos_net, oos_to, oos_excess=oos_ex)
        result["cost_sensitivity"] = compute_cost_sensitivity(
            dev_prices,
            None,
            folds,
            policy,
            per_fold_weights=per_fold_weights,
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
    *,
    fit_fn: FitFn | None = None,
) -> float | None:
    """Measure worst OOS Sharpe drop across neighboring parameter values (anti-curve-fitting check)."""
    if not neighbor_params_list:
        return None
    worst_drop = 0.0
    for alt_params in neighbor_params_list:
        res = evaluate_walk_forward(
            strategy_fn,
            dev_prices,
            alt_params,
            policy,
            fit_fn=fit_fn,
            include_diagnostics=False,
        )
        alt_sharpe = float(res["out_of_sample"]["sharpe"])
        drop = base_oos_sharpe - alt_sharpe
        worst_drop = max(worst_drop, drop)
    return round(worst_drop, 6)
