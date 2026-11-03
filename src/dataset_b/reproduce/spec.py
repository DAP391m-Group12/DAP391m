"""Accepted RQ1 run identities and the checks that bind a reproduction to them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .manifest import ReproduceError


METRIC_NAMES = (
    "roc_auc_approved",
    "average_precision_denied",
    "precision_denied",
    "recall_denied",
    "f1_denied",
    "recall_approved",
    "balanced_accuracy",
    "brier_approved",
    "calibration_intercept",
    "calibration_slope",
)
OUTPUT_DIRECTORY = "reports/reproducibility/rq1"
BOOTSTRAP_MANIFEST = "artifacts/dataset_b/rq1_test_bootstrap_manifest.json"
BOOTSTRAP_INDICES = "artifacts/dataset_b/rq1_test_bootstrap_indices.npz"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    scope: str
    run_id: str
    profile: str
    run_root: str
    checkpoint: str
    threshold_file: str
    test_directory: str
    requires_cuda: bool

    def path(self, relative: str) -> str:
        return f"{self.run_root}/{relative}"


MODELS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "logistic_regression", "lr", "lr_final_15f_20261002_01", "classical_cpu",
            "artifacts/dataset_b/lr/final/lr_final_15f_20261002_01",
            "checkpoints/fitted_pipeline.joblib", "validation/threshold_selection.json", "test", False,
        ),
        ModelSpec(
            "random_forest", "rf", "rf_final_15f_20261002_01", "classical_cpu",
            "artifacts/dataset_b/rf/final/rf_final_15f_20261002_01",
            "checkpoints/fitted_pipeline.joblib", "validation/threshold_selection.json", "test", False,
        ),
        ModelSpec(
            "xgboost", "xgboost", "xgb_final_15f_20261003_01", "classical_cpu",
            "artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01",
            "checkpoints/fitted_pipeline.joblib", "validation/threshold_selection.json", "test", False,
        ),
        ModelSpec(
            "plain_mlp", "plain_mlp", "plain_mlp_final_15f_20261004_03", "plain_mlp_cuda",
            "artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03",
            "full_train/checkpoints/fitted_bundle.joblib",
            "validation/threshold_selection/threshold_selection.json", "test_evaluation", True,
        ),
        ModelSpec(
            "realmlp", "realmlp", "realmlp_final_15f_20261004_01", "realmlp_cuda",
            "artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01",
            "full_train/fitted_model.joblib", "threshold/threshold_selection.json", "test_evaluation", True,
        ),
        ModelSpec(
            "tabmmini", "tabmmini", "tabmmini_final_15f_20261004_01", "tabmmini_cuda",
            "artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01",
            "full_train/model/inference_bundle.pt", "validation/threshold_selection.json", "test_evaluation", True,
        ),
    )
}


def check_run_identity(spec: ModelSpec, *payloads: Mapping[str, Any]) -> None:
    """Reject evidence that belongs to any run other than the accepted one."""
    for payload in payloads:
        if payload.get("run_id") != spec.run_id:
            raise ReproduceError(f"{spec.key}: run identity differs: {payload.get('run_id')!r}")


def check_threshold(spec: ModelSpec, selection: Mapping[str, Any], stored_metrics: Mapping[str, Any]) -> float:
    """Return the frozen Validation threshold, which Test evidence must also record."""
    selected = selection.get("selected_threshold")
    if isinstance(selected, bool) or not isinstance(selected, (int, float)) or not 0 < selected < 1:
        raise ReproduceError(f"{spec.key}: frozen threshold is invalid: {selected!r}")
    if stored_metrics.get("threshold") != selected:
        raise ReproduceError(
            f"{spec.key}: threshold differs: frozen {selected!r}, stored {stored_metrics.get('threshold')!r}"
        )
    return float(selected)


def check_record_order(spec: ModelSpec, observed: Sequence[str], expected: Sequence[str]) -> None:
    """Require the same Test records in the same order."""
    if len(observed) != len(expected):
        raise ReproduceError(f"{spec.key}: Test row count differs: {len(observed)} and {len(expected)}")
    if len(set(observed)) != len(observed):
        raise ReproduceError(f"{spec.key}: Test record IDs are not unique")
    for position, (left, right) in enumerate(zip(observed, expected)):
        if left != right:
            raise ReproduceError(f"{spec.key}: Test record order differs at row {position}: {left} and {right}")


def check_split_digest(spec: ModelSpec, observed_digest: str, bootstrap_manifest: Mapping[str, Any]) -> None:
    """Require the Test IDs the locked bootstrap indices were built for."""
    expected = bootstrap_manifest.get("test_record_id_sha256")
    if observed_digest != expected:
        raise ReproduceError(f"{spec.key}: Test split differs from the locked bootstrap manifest")
