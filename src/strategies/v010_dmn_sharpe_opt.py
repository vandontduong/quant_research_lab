"""Batch 1 / Cycle 10 (inspired by Deep Momentum Networks, arXiv:1904.04912).

Stylized, paper-inspired adaptation on the 6-ETF daily universe (not a full reproduction
of the paper's neural-network architecture; Lim, Zohren & Roberts, 2019):
- Computes causal multi-horizon volatility-scaled trend features (21d, 63d, 126d, 252d).
- On each monthly rebalance bar `t`, fits a 4-parameter linear feature weighting `theta`
  via deterministic projected coordinate ascent on a trailing realized Sharpe-ratio objective
  with L2 regularization (no deep sequence model).
- Maps the linear score through `tanh` into volatility-scaled long-only ETF weights (with
  defensive SHY absorbing unallocated risk budget).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.v001_inv_vol import _cap_and_normalize

STRATEGY_NAME = "v010_dmn_sharpe_opt"
STRATEGY_NOTE = "DMN-inspired (arXiv:1904.04912): stylized 4-feature Sharpe-ascent linear tilt on inv-vol (no neural net)."
DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 63,
    "train_window": 504,
    "rebalance_every_days": 21,
    "l2_reg": 0.50,
    "score_scale": 0.40,
    "min_vol_floor": 0.04,
    "max_weight": 0.45,
}
NEIGHBOR_PARAMS: list[dict[str, Any]] = [
    {**DEFAULT_PARAMS, "train_window": 378},
    {**DEFAULT_PARAMS, "train_window": 756},
    {**DEFAULT_PARAMS, "score_scale": 0.25},
    {**DEFAULT_PARAMS, "score_scale": 0.60},
]
HYPOTHESIS = (
    "Deep Momentum Networks (arXiv:1904.04912): optimizing multi-horizon vol-scaled trend "
    "weights directly on trailing realized Sharpe (with L2 regularization) rather than return "
    "error should adaptively size trend tilts on top of inverse-volatility risk parity."
)


def _compute_vol_scaled_features(prices: pd.DataFrame, vol_window: int) -> np.ndarray:
    """Return strictly causal (n_days, n_assets, 4) normalized trend features."""
    rets = prices.pct_change().fillna(0.0)
    vol = rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    vol_arr = np.maximum(vol.to_numpy(dtype=float), 0.04)
    feats = []
    for h in (21, 63, 126, 252):
        mom = prices.pct_change(h).to_numpy(dtype=float)
        scaled = np.tanh(np.nan_to_num(mom, nan=0.0) / (vol_arr * np.sqrt(h / 252.0)))
        feats.append(scaled)
    return np.stack(feats, axis=-1)


def _optimize_sharpe_theta(
    feat_win: np.ndarray,
    next_ret_win: np.ndarray,
    inv_vol_win: np.ndarray,
    l2_reg: float,
) -> np.ndarray:
    """Deterministic causal gradient ascent on trailing Sharpe objective (4 feature weights)."""
    # feat_win: (T, N, K), next_ret_win: (T, N), inv_vol_win: (T, N)
    k = feat_win.shape[2]
    theta = np.full(k, 1.0 / k, dtype=float)
    lr = 0.15
    for _ in range(15):
        raw_score = np.tensordot(feat_win, theta, axes=([2], [0]))  # (T, N)
        # Soft positive sizing around baseline inverse-vol
        tilt = 1.0 + 0.5 * np.tanh(raw_score)
        w = inv_vol_win * tilt
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        port_r = (w * next_ret_win).sum(axis=1)
        mu = float(port_r.mean())
        sd = float(port_r.std(ddof=1)) + 1e-8
        # Finite-difference gradient on regularized Sharpe (4 dims -> fast & exact)
        base_obj = (mu / sd) - l2_reg * float(np.dot(theta, theta))
        grad = np.zeros(k, dtype=float)
        eps = 1e-3
        for j in range(k):
            t_eps = theta.copy()
            t_eps[j] += eps
            rs = np.tensordot(feat_win, t_eps, axes=([2], [0]))
            w_e = inv_vol_win * (1.0 + 0.5 * np.tanh(rs))
            w_e = w_e / np.maximum(w_e.sum(axis=1, keepdims=True), 1e-12)
            pr = (w_e * next_ret_win).sum(axis=1)
            obj_e = (float(pr.mean()) / (float(pr.std(ddof=1)) + 1e-8)) - l2_reg * float(np.dot(t_eps, t_eps))
            grad[j] = (obj_e - base_obj) / eps
        theta = np.clip(theta + lr * grad, -1.5, 1.5)
    return theta


def generate_weights(prices: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return strictly causal DMN Sharpe-optimized target weights."""
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    vol_window = int(cfg["vol_window"])
    train_window = int(cfg["train_window"])
    step = int(cfg["rebalance_every_days"])
    l2_reg = float(cfg["l2_reg"])
    score_scale = float(cfg["score_scale"])
    min_vol_floor = float(cfg["min_vol_floor"])
    max_weight = float(cfg["max_weight"])

    n_days, n_assets = prices.shape
    rets = prices.pct_change().fillna(0.0)
    ret_arr = rets.to_numpy(dtype=float)
    rolling_vol = (rets.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)).to_numpy(
        dtype=float
    )
    inv_vol_all = 1.0 / np.maximum(np.nan_to_num(rolling_vol, nan=0.15), min_vol_floor)
    feats = _compute_vol_scaled_features(prices, vol_window)

    w_arr = np.zeros((n_days, n_assets), dtype=float)
    current = np.full(n_assets, 1.0 / n_assets, dtype=float)
    min_bars = max(252, vol_window)

    for i in range(n_days):
        if i % step == 0:
            vols = rolling_vol[i]
            if i >= min_bars and np.all(np.isfinite(vols)):
                floored = np.maximum(vols, min_vol_floor)
                if i >= min_bars + 63:
                    start = max(min_bars, i - train_window)
                    # Strictly causal pairs: features at t in [start, i-1] predict returns at t+1 <= i.
                    # Subsample every 5 bars inside training window for fast, low-noise Sharpe updates.
                    idx_slice = slice(start, i, 5)
                    next_slice = slice(start + 1, i + 1, 5)
                    theta = _optimize_sharpe_theta(
                        feats[idx_slice],
                        ret_arr[next_slice],
                        inv_vol_all[idx_slice],
                        l2_reg,
                    )
                else:
                    theta = np.full(feats.shape[2], 0.25, dtype=float)
                score = feats[i] @ theta
                tilt = np.exp(score_scale * np.tanh(score))
                raw = (1.0 / floored) * tilt
                current = _cap_and_normalize(raw, max_weight)
            else:
                current = np.full(n_assets, 1.0 / n_assets, dtype=float)
        else:
            grown = current * (1.0 + ret_arr[i])
            total = float(grown.sum())
            current = grown / total if total > 0 else np.full(n_assets, 1.0 / n_assets, dtype=float)
            if float(current.max()) > 0.495:
                current = _cap_and_normalize(current, 0.49)
        w_arr[i] = current

    return pd.DataFrame(w_arr, index=prices.index, columns=prices.columns)
