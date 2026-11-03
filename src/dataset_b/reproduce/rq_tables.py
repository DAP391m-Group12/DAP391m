"""Compare two CSV tables cell by cell: exact for text, absolute tolerance for numbers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


FIELDS = ("model", "metric", "expected", "observed", "delta", "tolerance", "status", "source")


def comparison_row(model: str, metric: str, expected: Any, observed: Any, delta: Any, tolerance: Any, passed: bool, source: str) -> dict[str, Any]:
    return {
        "model": model,
        "metric": metric,
        "expected": expected,
        "observed": observed,
        "delta": delta,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "source": source,
    }


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _cells_match(left: str, right: str, atol: float) -> tuple[bool, float]:
    if left == right:
        return True, 0.0
    first, second = _number(left), _number(right)
    if first is None or second is None:
        return False, math.inf
    if not math.isfinite(first) or not math.isfinite(second):
        same = (math.isnan(first) and math.isnan(second)) or first == second
        return same, 0.0 if same else math.inf
    delta = abs(first - second)
    return delta <= atol, delta


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


def compare_csv(
    route: str,
    name: str,
    expected: Path,
    observed: Path,
    atol: float,
    *,
    ignore_columns: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return one row per table; shape, column, and text differences all fail."""
    source = expected.as_posix()
    if not expected.is_file() or not observed.is_file():
        return comparison_row(route, name, expected.is_file(), observed.is_file(), None, atol, False, source)
    left_header, left = _read(expected)
    right_header, right = _read(observed)
    if left_header != right_header:
        return comparison_row(route, name, left_header, right_header, None, "exact columns", False, source)
    if len(left) != len(right):
        return comparison_row(route, name, len(left), len(right), None, "exact rows", False, source)
    keep = [index for index, column in enumerate(left_header) if column not in ignore_columns]
    worst = 0.0
    failures = 0
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            failures += 1
            continue
        for index in keep:
            matched, delta = _cells_match(left_row[index], right_row[index], atol)
            failures += not matched
            worst = max(worst, delta)
    return comparison_row(route, name, len(left), len(right), worst, atol, failures == 0, source)
