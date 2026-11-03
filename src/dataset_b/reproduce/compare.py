"""Compare reproduction outputs with accepted references under the locked policy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import ReproduceError
from .spec import METRIC_NAMES, MODELS, OUTPUT_DIRECTORY, ModelSpec


POLICY_PATH = "config/dataset_b/reproducibility/numerical_policy.json"
FIELDS = ("model", "metric", "expected", "observed", "delta", "tolerance", "status", "source")
SLOPE = "calibration_slope"


def load_policy(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((project_root / POLICY_PATH).read_text(encoding="utf-8"))
    policy = {}
    for key in MODELS:
        if key not in payload.get("models", {}):
            raise ReproduceError(f"Numerical policy has no entry for {key}")
        policy[key] = {**payload["default"], **payload["models"][key]}
    return policy


def values_match(expected: float | None, observed: float | None, atol: float) -> tuple[bool, float | None]:
    """Compare two floats; a missing or non-finite value matches only the same state."""
    if expected is None or observed is None:
        return expected is None and observed is None, None
    if not math.isfinite(expected) or not math.isfinite(observed):
        same = (math.isnan(expected) and math.isnan(observed)) or expected == observed
        return same, None
    delta = abs(observed - expected)
    return delta <= atol, delta


def slope_match(expected: float, observed: float, decimals: int) -> tuple[bool, float | None]:
    """Apply the locked six-decimal rule to calibration slope values."""
    if not math.isfinite(expected) or not math.isfinite(observed):
        return values_match(expected, observed, 0.0)
    return round(expected, decimals) == round(observed, decimals), abs(observed - expected)


def _row(model: str, metric: str, expected: Any, observed: Any, delta: Any, tolerance: Any, passed: bool, source: str) -> dict[str, Any]:
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def compare_point_metrics(key: str, expected: Mapping[str, Any], observed: Mapping[str, Any], policy: Mapping[str, Any], source: str) -> Iterable[dict[str, Any]]:
    for name in METRIC_NAMES:
        if name not in expected or name not in observed:
            yield _row(key, f"point:{name}", expected.get(name), observed.get(name), None, policy["point_atol"], False, source)
            continue
        matched, delta = values_match(_number(expected[name]), _number(observed[name]), policy["point_atol"])
        yield _row(key, f"point:{name}", expected[name], observed[name], delta, policy["point_atol"], matched, source)


def compare_intervals(key: str, expected: Mapping[str, Any], observed: Mapping[str, Any], policy: Mapping[str, Any], source: str) -> Iterable[dict[str, Any]]:
    for name in METRIC_NAMES:
        left, right = expected.get(name), observed.get(name)
        if not left or not right:
            yield _row(key, f"ci:{name}", left, right, None, policy["ci_atol"], False, source)
            continue
        for endpoint in ("lower_2_5", "upper_97_5"):
            matched, delta = values_match(_number(left.get(endpoint)), _number(right.get(endpoint)), policy["ci_atol"])
            yield _row(key, f"ci:{name}:{endpoint}", left.get(endpoint), right.get(endpoint), delta, policy["ci_atol"], matched, source)
        for count in ("valid_replicates", "invalid_replicates"):
            yield _row(key, f"ci:{name}:{count}", left.get(count), right.get(count), None, "exact", left.get(count) == right.get(count), source)


def compare_bootstrap(key: str, expected: list[Mapping[str, str]], observed: list[Mapping[str, str]], policy: Mapping[str, Any], source: str) -> Iterable[dict[str, Any]]:
    same_ids = [row.get("replicate_id") for row in expected] == [row.get("replicate_id") for row in observed]
    yield _row(key, "bootstrap:replicate_ids", len(expected), len(observed), None, "exact", same_ids and len(expected) == 1000, source)
    if not same_ids:
        return
    payload_differences = sum(left.get("invalid_metrics") != right.get("invalid_metrics") for left, right in zip(expected, observed))
    yield _row(key, "bootstrap:invalid_payload", 0, payload_differences, None, "exact", payload_differences == 0, source)
    for name in METRIC_NAMES:
        worst: float = 0.0
        failures = 0
        for left, right in zip(expected, observed):
            if name not in left or name not in right:
                failures += 1
                continue
            first, second = float(left[name]), float(right[name])
            if name == SLOPE:
                matched, delta = slope_match(first, second, policy["calibration_slope_decimals"])
            else:
                matched, delta = values_match(first, second, policy["bootstrap_atol"])
            failures += not matched
            worst = max(worst, delta or 0.0)
        tolerance = f"round({policy['calibration_slope_decimals']})" if name == SLOPE else policy["bootstrap_atol"]
        yield _row(key, f"bootstrap:{name}", 0, failures, worst, tolerance, failures == 0, source)


def compare_predictions(key: str, expected: list[Mapping[str, str]], observed: list[Mapping[str, str]], policy: Mapping[str, Any], source: str) -> Iterable[dict[str, Any]]:
    same_ids = [row["record_id"] for row in expected] == [row["record_id"] for row in observed]
    same_labels = [row["y_approved"] for row in expected] == [row["y_approved"] for row in observed]
    yield _row(key, "prediction:record_ids", len(expected), len(observed), None, "exact", same_ids, source)
    yield _row(key, "prediction:labels", len(expected), len(observed), None, "exact", same_labels, source)
    if not same_ids:
        return
    worst = max(abs(float(left["p_approved"]) - float(right["p_approved"])) for left, right in zip(expected, observed))
    yield _row(key, "prediction:p_approved_max_abs", 0, worst, worst, policy["prediction_atol"], worst <= policy["prediction_atol"], source)


def output_binding_problems(output: Path) -> list[str]:
    """Reject outputs that no checkpoint-based receipt produced, such as copied stored files."""
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        return ["receipt is missing"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    problems = []
    if receipt.get("classification") != "checkpoint-based":
        problems.append(f"classification is {receipt.get('classification')!r}")
    hashes = receipt.get("output_hashes", {})
    if "rq1_test_metrics.json" not in hashes:
        problems.append("receipt does not cover the metrics file")
    for name, expected in hashes.items():
        path = output / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"{name} differs from its receipt")
    return problems


def compare_model(project_root: Path, spec: ModelSpec, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = project_root / spec.run_root / spec.test_directory
    output = project_root / OUTPUT_DIRECTORY / spec.key
    source = f"{spec.run_root}/{spec.test_directory}"
    metrics_path = output / "rq1_test_metrics.json"
    if not metrics_path.is_file():
        return [_row(spec.key, "output", "present", "missing", None, "exact", False, source)]
    problems = output_binding_problems(output)
    if problems:
        return [_row(spec.key, "output_receipt", "bound", "; ".join(problems), None, "exact", False, source)]
    expected = json.loads((reference / "rq1_test_metrics.json").read_text(encoding="utf-8"))
    observed = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [
        _row(spec.key, "run_id", spec.run_id, observed.get("run_id"), None, "exact", observed.get("run_id") == spec.run_id, source),
        _row(spec.key, "threshold", expected.get("threshold"), observed.get("threshold"), None, "exact", expected.get("threshold") == observed.get("threshold"), source),
    ]
    rows += compare_point_metrics(spec.key, expected.get("point_metrics", {}), observed.get("point_metrics", {}), policy, source)
    rows += compare_intervals(spec.key, expected.get("confidence_intervals_95_percentile", {}), observed.get("confidence_intervals_95_percentile", {}), policy, source)
    rows += compare_bootstrap(spec.key, _read_rows(reference / "rq1_bootstrap_metrics.csv"), _read_rows(output / "rq1_bootstrap_metrics.csv"), policy, source)
    rows += compare_predictions(spec.key, _read_rows(reference / "test_predictions.csv"), _read_rows(output / "test_predictions.csv"), policy, source)
    return rows


def compare_all(project_root: Path) -> dict[str, Any]:
    """Write the comparison table; any missing or out-of-tolerance row fails the gate."""
    policy = load_policy(project_root)
    rows: list[dict[str, Any]] = []
    for key, spec in MODELS.items():
        rows += compare_model(project_root, spec, policy[key])
    directory = project_root / OUTPUT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / "comparison.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    return {
        "status": "FAIL" if failed else "PASS",
        "row_count": len(rows),
        "failed_count": len(failed),
        "failed": [{"model": row["model"], "metric": row["metric"], "delta": row["delta"]} for row in failed[:40]],
        "table": table.relative_to(project_root).as_posix(),
    }
