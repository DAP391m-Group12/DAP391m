"""Cached-score Equalized-Odds post-processing for Dataset B RQ4a.

The frozen base models are represented only by stored `p_approved` values; no
model is loaded, fitted, or run. Persistence follows the interface locked by the
accepted T3 smoke: `joblib.dump(compress=3)` with the adapter defined in this
importable module, plus a `policy_interpolation.json` audit export.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

from src.dataset_b.mitigation.inputs import LOCKED_POSTPROCESSOR


POLICY_FILENAME = "policy.joblib"
INTERPOLATION_FILENAME = "policy_interpolation.json"


class RQ4APolicyError(RuntimeError):
    """Raised when post-processing would silently fall back or break the locked contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ4APolicyError(message)


class CachedProbabilityEstimator(ClassifierMixin, BaseEstimator):
    """Frozen `p_approved` looked up by `record_id`; satisfies `prefit=True` without fitting.

    The lookup must have unique `record_id`. Query rows may repeat a `record_id`
    (with-replacement bootstrap); one output row is returned per query row, in
    query order. Unknown or missing query IDs fail.
    """

    def __init__(self, p_approved: pd.Series):
        self.p_approved = p_approved
        _require(isinstance(p_approved, pd.Series), "Cached probabilities must be a pandas Series indexed by record_id.")
        _require(p_approved.index.notna().all(), "Missing record_id in cached probabilities.")
        normalized_index = p_approved.index.astype(str)
        _require(normalized_index.is_unique, "Duplicate record_id in cached probabilities (after string normalization).")
        values = p_approved.to_numpy(dtype=np.float64)
        _require(np.isfinite(values).all(), "Cached p_approved contains non-finite values.")
        _require(np.all((values >= 0.0) & (values <= 1.0)), "Cached p_approved is outside [0, 1].")
        self._p = pd.Series(values, index=normalized_index)
        self.classes_ = np.array([0, 1])
        self.predict_proba_calls = 0
        self.fit_calls = 0

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise RQ4APolicyError("CachedProbabilityEstimator must never be fitted.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _require(isinstance(X, pd.DataFrame) and "record_id" in X.columns, "X must be a DataFrame with a record_id column.")
        raw_ids = X["record_id"]
        _require(raw_ids.notna().all(), "Missing record_id in X.")
        ids = raw_ids.astype(str)
        missing = ~ids.isin(self._p.index)
        _require(not missing.any(), f"Unknown record_id in X: {int(missing.sum())} rows.")
        self.predict_proba_calls += 1
        p = self._p.reindex(ids.to_numpy()).to_numpy()
        _require(len(p) == len(ids), "Cached lookup must return one row per query occurrence.")
        return np.column_stack([1.0 - p, p])


def baseline_decisions(p_approved: Sequence[float] | np.ndarray, threshold: float) -> np.ndarray:
    """Frozen baseline hard prediction: Approved when p_approved >= frozen threshold."""

    p = np.asarray(p_approved, dtype=np.float64)
    _require(np.isfinite(threshold) and 0.0 <= threshold <= 1.0, "Frozen threshold must be finite within [0, 1].")
    return (p >= threshold).astype(np.int8)


def decision_random_state(seed: int) -> np.random.RandomState:
    """A fresh RandomState per decision draw; Fairlearn advances a passed instance in place."""

    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool), "Decision seed must be an integer.")
    _require(0 <= int(seed) < 2**32, "Decision seed must be within [0, 2**32).")
    return np.random.RandomState(int(seed))


def _frame(record_ids: Sequence[str] | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"record_id": np.asarray(record_ids).astype(str)})


def assert_groups_fitted(policy: ThresholdOptimizer, groups: Sequence[str] | np.ndarray) -> None:
    """Fairlearn assigns positive probability 0 to unseen groups; fail instead."""

    fitted = set(policy.interpolated_thresholder_.interpolation_dict)
    unseen = set(np.asarray(groups).astype(str)) - {str(g) for g in fitted}
    _require(not unseen, f"Groups not fitted by the policy: {sorted(unseen)}")


def fit_policy(
    adapter: CachedProbabilityEstimator,
    record_ids: Sequence[str] | np.ndarray,
    y_approved: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    roster: Sequence[str],
    *,
    params: Mapping[str, Any] = LOCKED_POSTPROCESSOR,
) -> ThresholdOptimizer:
    """Fit ThresholdOptimizer on the roster rows of the given partition only."""

    _require(dict(params) == LOCKED_POSTPROCESSOR, "ThresholdOptimizer parameters differ from the locked contract.")
    ids = np.asarray(record_ids).astype(str)
    y = np.asarray(y_approved)
    labels = np.asarray(groups).astype(str)
    _require(len(ids) == len(y) == len(labels), "Fit inputs must have equal lengths.")
    _require(np.isin(y, [0, 1]).all(), "y_approved must be binary.")
    mask = np.isin(labels, list(roster))
    _require(set(labels[mask]) == set(roster), "Every roster group must be present in the fit rows.")
    policy = ThresholdOptimizer(estimator=adapter, **dict(params))
    fit_calls_before = adapter.fit_calls
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy.fit(_frame(ids[mask]), pd.Series(y[mask].astype(int)), sensitive_features=pd.Series(labels[mask]))
    _require(adapter.fit_calls == fit_calls_before and policy.estimator_ is adapter, "Base adapter must not be fitted.")
    _require(
        {str(g) for g in policy.interpolated_thresholder_.interpolation_dict} == set(roster),
        "Fitted policy groups differ from the roster.",
    )
    return policy


def apply_policy(
    policy: ThresholdOptimizer,
    record_ids: Sequence[str] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    baseline: Sequence[int] | np.ndarray,
    roster: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mitigated decisions for roster rows; frozen baseline outside the roster.

    Rows may repeat a `record_id` (locked with-replacement Test indices); every
    occurrence receives its own independent draw, taken in the order of the
    roster rows as given from a fresh RandomState(seed). The same rows in the
    same order with the same seed reproduce the decisions bit for bit.
    Returns (decisions, in_roster mask), both aligned with the input rows.
    """

    ids = np.asarray(record_ids).astype(str)
    labels = np.asarray(groups).astype(str)
    base = np.asarray(baseline)
    _require(len(ids) == len(labels) == len(base), "Apply inputs must have equal lengths.")
    _require(np.isin(base, [0, 1]).all(), "Baseline decisions must be binary.")
    in_roster = np.isin(labels, list(roster))
    decisions = base.astype(np.int8).copy()
    if in_roster.any():
        assert_groups_fitted(policy, labels[in_roster])
        mitigated = policy.predict(
            _frame(ids[in_roster]),
            sensitive_features=pd.Series(labels[in_roster]),
            random_state=decision_random_state(seed),
        )
        decisions[in_roster] = np.asarray(mitigated, dtype=np.int8)
    return decisions, in_roster


def interpolation_summary(policy: ThresholdOptimizer) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group, b in sorted(policy.interpolated_thresholder_.interpolation_dict.items()):
        entry: dict[str, Any] = {
            "p0": float(b.p0),
            "operation0": {"operator": b.operation0.operator, "threshold": float(b.operation0.threshold)},
            "p1": float(b.p1),
            "operation1": {"operator": b.operation1.operator, "threshold": float(b.operation1.threshold)},
        }
        if "p_ignore" in b:
            entry["p_ignore"] = float(b.p_ignore)
            entry["prediction_constant"] = float(b.prediction_constant)
        summary[str(group)] = entry
    return summary


def save_policy(policy: ThresholdOptimizer, directory: Path) -> dict[str, str]:
    """Write policy.joblib and policy_interpolation.json; never overwrite. Returns SHA-256 by file."""

    directory = Path(directory)
    targets = [directory / POLICY_FILENAME, directory / INTERPOLATION_FILENAME]
    _require(not any(t.exists() for t in targets), f"Refusing to overwrite an existing policy in {directory}.")
    _require(
        type(policy.estimator_).__module__ == __name__,
        "Policy adapter must be the importable mitigation adapter, not a __main__ class.",
    )
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, targets[0], compress=3)
    targets[1].write_text(json.dumps(interpolation_summary(policy), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {t.name: hashlib.sha256(t.read_bytes()).hexdigest() for t in targets}


def load_policy(directory: Path) -> ThresholdOptimizer:
    policy = joblib.load(Path(directory) / POLICY_FILENAME)
    _require(isinstance(policy, ThresholdOptimizer), "Persisted object is not a ThresholdOptimizer.")
    _require(isinstance(policy.estimator_, CachedProbabilityEstimator), "Persisted policy does not use the cached adapter.")
    return policy
