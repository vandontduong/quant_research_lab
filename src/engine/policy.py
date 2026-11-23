"""Immutable risk policy loader, SHA-256 verification, and promotion gate evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskPolicy:
    raw: dict[str, Any]
    sha256: str
    path: Path

    @property
    def universe(self) -> list[str]:
        return [str(t).upper() for t in self.raw["universe"]]

    @property
    def benchmark_ticker(self) -> str:
        return str(self.raw["benchmark_ticker"]).upper()

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw["execution"]

    @property
    def risk_gates(self) -> dict[str, Any]:
        return self.raw["risk_gates"]

    @property
    def promotion(self) -> dict[str, Any]:
        return self.raw["promotion"]


def load_policy(path: str | Path = "config/risk_policy.json") -> RiskPolicy:
    """Load config/risk_policy.json and compute its SHA-256 integrity hash."""
    p = Path(path)
    raw_bytes = p.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    if not data.get("research_only", False) or data.get("allow_live_execution", False):
        raise ValueError("Policy violation: live execution is forbidden in this research sandbox.")
    return RiskPolicy(raw=data, sha256=digest, path=p)


def evaluate_gates(
    policy: RiskPolicy,
    *,
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    perturbation_drop: float | None = None,
    champion_oos_sharpe: float | None = None,
    is_baseline: bool = False,
) -> dict[str, Any]:
    """Check all hard risk ceilings and promotion criteria without weakening any rule."""
    gates = policy.risk_gates
    promo = policy.promotion
    checks: dict[str, dict[str, Any]] = {}

    oos_dd = float(oos_metrics["max_drawdown"])
    max_dd_limit = float(gates["max_oos_drawdown"])
    checks["max_oos_drawdown"] = {
        "passed": oos_dd >= max_dd_limit,
        "actual": round(oos_dd, 6),
        "limit": max_dd_limit,
    }

    ann_turnover = float(oos_metrics["annual_turnover"])
    max_to_limit = float(gates["max_annual_turnover"])
    checks["max_annual_turnover"] = {
        "passed": ann_turnover <= max_to_limit,
        "actual": round(ann_turnover, 6),
        "limit": max_to_limit,
    }

    rebal_events = int(oos_metrics["rebalance_events"])
    min_rebal = int(gates["min_rebalance_events"])
    checks["min_rebalance_events"] = {
        "passed": rebal_events >= min_rebal,
        "actual": rebal_events,
        "limit": min_rebal,
    }

    is_sharpe = float(is_metrics["sharpe"])
    oos_sharpe = float(oos_metrics["sharpe"])
    degradation = is_sharpe - oos_sharpe
    max_deg = float(gates["max_is_to_oos_sharpe_degradation"])
    checks["is_to_oos_sharpe_degradation"] = {
        "passed": degradation <= max_deg,
        "actual": round(degradation, 6),
        "limit": max_deg,
    }

    if perturbation_drop is not None:
        max_pdrop = float(gates["max_perturbation_sharpe_drop"])
        checks["perturbation_stability"] = {
            "passed": perturbation_drop <= max_pdrop,
            "actual": round(perturbation_drop, 6),
            "limit": max_pdrop,
        }

    risk_passed = all(c["passed"] for c in checks.values())

    if is_baseline or champion_oos_sharpe is None:
        improvement = None
        beats_champion = True
    else:
        min_imp = float(promo["min_oos_sharpe_improvement"])
        improvement = oos_sharpe - champion_oos_sharpe
        beats_champion = improvement >= min_imp
        checks["beats_champion_margin"] = {
            "passed": beats_champion,
            "actual": round(improvement, 6),
            "limit": min_imp,
        }

    promote = risk_passed and beats_champion
    failed_gates = [name for name, c in checks.items() if not c["passed"]]
    return {
        "promote": promote,
        "risk_passed": risk_passed,
        "beats_champion": beats_champion,
        "oos_sharpe_improvement": round(improvement, 6) if improvement is not None else None,
        "failed_gates": failed_gates,
        "checks": checks,
    }
