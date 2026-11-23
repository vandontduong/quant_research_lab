"""Lightweight AST & parameter complexity auditor inspired by autoresearch's simplicity criterion."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.IfExp,
    ast.Try,
    ast.BoolOp,
)


def analyze_strategy_complexity(
    strategy_file: str | Path,
    params: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Compute deterministic SLOC, parameter count, branch count, and function count for a strategy."""
    path = Path(strategy_file)
    if not path.exists():
        return {
            "sloc": 0,
            "param_count": len(params) if params is not None else 0,
            "branch_count": 0,
            "function_count": 0,
        }

    source = path.read_text(encoding="utf-8")
    sloc = sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "sloc": sloc,
            "param_count": len(params) if params is not None else 0,
            "branch_count": 0,
            "function_count": 0,
        }

    branch_count = sum(1 for node in ast.walk(tree) if isinstance(node, _BRANCH_NODES))
    function_count = sum(
        1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    if params is not None:
        param_count = len(params)
    else:
        param_count = _extract_default_params_count(tree)

    return {
        "sloc": sloc,
        "param_count": param_count,
        "branch_count": branch_count,
        "function_count": function_count,
    }


def _extract_default_params_count(tree: ast.AST) -> int:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "DEFAULT_PARAMS"
                    and isinstance(node.value, ast.Dict)
                ):
                    return len(node.value.keys)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "DEFAULT_PARAMS"
            and isinstance(node.value, ast.Dict)
        ):
            return len(node.value.keys)
    return 0
