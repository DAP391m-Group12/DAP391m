"""Read-only predictor adapters for the six frozen Dataset B RQ2 models.

The adapters expose one operation only: map the ordered final 15-feature frame
to a ``float64`` vector of ``p_approved`` values.  They never fit a model,
select a threshold, or write an artifact.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from src.utils.path import PROJECT_ROOT


EXPECTED_FEATURE_VERSION = "feature_set_v1"
EXPECTED_SPLIT_VERSION = "split_v1"
PROBABILITY_SUM_ATOL = 1.0e-6
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "dataset_b"
FEATURE_COLUMNS = (
    "loan_type",
    "loan_purpose",
    "lien_status",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "income",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "occupancy_type",
    "preapproval",
    "interest_only_payment",
    "balloon_payment",
)
EXEMPTION_FLAG_COLUMNS = (
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "interest_only_payment",
    "balloon_payment",
)
_CATEGORICAL_DOMAINS = {
    "loan_type": frozenset({"1", "2", "3", "4"}),
    "loan_purpose": frozenset({"1", "2", "31", "32", "4", "5"}),
    "lien_status": frozenset({"1", "2"}),
    "reverse_mortgage": frozenset({"1", "2", "1111", "Exempt"}),
    "open-end_line_of_credit": frozenset({"1", "2", "1111", "Exempt"}),
    "business_or_commercial_purpose": frozenset({"1", "2", "1111", "Exempt"}),
    "occupancy_type": frozenset({"1", "2", "3"}),
    "preapproval": frozenset({"1", "2"}),
    "interest_only_payment": frozenset({"1", "2", "1111", "Exempt"}),
    "balloon_payment": frozenset({"1", "2", "1111", "Exempt"}),
}
_DTI_DOMAIN = frozenset(
    {"<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%", "NA", "Exempt"}
    | {str(value) for value in range(36, 50)}
)
_NUMERIC_SPECIAL = {
    "income": frozenset({"NA"}),
    "loan_amount": frozenset(),
    "loan_to_value_ratio": frozenset({"NA", "Exempt"}),
    "loan_term": frozenset({"NA", "Exempt"}),
}


class FrozenPredictorError(RuntimeError):
    """Raised when a frozen bundle or prediction violates the RQ2 boundary."""


class FrozenPredictor(Protocol):
    """Common RQ2 predictor interface."""

    model_key: str
    run_id: str
    execution_device: str

    def predict_p_approved(self, X: pd.DataFrame) -> np.ndarray:
        """Return one finite ``float64`` approval probability per input row."""

    def predict_p_approved_timed(self, X: pd.DataFrame) -> "TimedPrediction":
        """Return probabilities with disjoint preprocessing and inference times."""

    def smoke_compare_stored(
        self, validation_rows: pd.DataFrame, *, atol: float
    ) -> None:
        """Fail unless fresh Validation probabilities match the frozen stored values."""


@dataclass(frozen=True)
class TimedPrediction:
    probabilities: np.ndarray
    preprocessing_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class FrozenPredictorSpec:
    model_key: str
    run_id: str
    run_directory: Path
    bundle_path: Path


_RUNS = {
    "lr": (
        "lr_final_15f_20261002_01",
        "lr/final/lr_final_15f_20261002_01",
        "checkpoints/fitted_pipeline.joblib",
    ),
    "rf": (
        "rf_final_15f_20261002_01",
        "rf/final/rf_final_15f_20261002_01",
        "checkpoints/fitted_pipeline.joblib",
    ),
    "xgboost": (
        "xgb_final_15f_20261003_01",
        "xgboost/final/xgb_final_15f_20261003_01",
        "checkpoints/fitted_pipeline.joblib",
    ),
    "plain_mlp": (
        "plain_mlp_final_15f_20261004_03",
        "plain_mlp/final/plain_mlp_final_15f_20261004_03",
        "full_train/checkpoints/fitted_bundle.joblib",
    ),
    "realmlp": (
        "realmlp_final_15f_20261004_01",
        "realmlp/final/realmlp_final_15f_20261004_01",
        "full_train/fitted_model.joblib",
    ),
    "tabmmini": (
        "tabmmini_final_15f_20261004_01",
        "tabmmini/final/tabmmini_final_15f_20261004_01",
        "full_train/model/inference_bundle.pt",
    ),
}


def available_model_keys() -> tuple[str, ...]:
    """Return the protocol-ordered predictor keys."""

    return tuple(_RUNS)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenPredictorError(f"Cannot read frozen predictor metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise FrozenPredictorError(f"Frozen predictor metadata must be an object: {path}")
    return payload


def _predictor_spec(model_key: str, artifact_root: str | Path) -> FrozenPredictorSpec:
    if model_key not in _RUNS:
        raise FrozenPredictorError(
            f"Unknown RQ2 model key {model_key!r}; expected one of {available_model_keys()}."
        )
    run_id, relative_root, relative_bundle = _RUNS[model_key]
    run_directory = Path(artifact_root) / relative_root
    return FrozenPredictorSpec(
        model_key=model_key,
        run_id=run_id,
        run_directory=run_directory,
        bundle_path=run_directory / relative_bundle,
    )


def _preflight(spec: FrozenPredictorSpec) -> None:
    frozen = spec.run_directory / "FROZEN"
    summary_path = spec.run_directory / "reporting" / "model_summary.json"
    if not frozen.is_file() or frozen.stat().st_size == 0:
        raise FrozenPredictorError(f"Frozen marker is missing or empty for {spec.model_key}.")
    if not spec.bundle_path.is_file() or spec.bundle_path.stat().st_size == 0:
        raise FrozenPredictorError(f"Inference bundle is missing or empty for {spec.model_key}.")

    summary = _read_json(summary_path)
    expected = {
        "run_id": spec.run_id,
        "model_key": spec.model_key,
        "reporting_status": "COMPLETE",
        "feature_version": EXPECTED_FEATURE_VERSION,
        "split_version": EXPECTED_SPLIT_VERSION,
    }
    observed = {name: summary.get(name) for name in expected}
    if observed != expected:
        raise FrozenPredictorError(
            f"Frozen reporting metadata differs for {spec.model_key}: {observed!r}."
        )


def _canonical_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise FrozenPredictorError("RQ2 predictor input must be a pandas DataFrame.")
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise FrozenPredictorError("RQ2 predictor input must use the ordered final 15 fields.")
    if len(X) == 0:
        raise FrozenPredictorError("RQ2 predictor input must contain at least one row.")

    cleaned = X.copy(deep=True)
    for column in FEATURE_COLUMNS:
        converted: list[object] = []
        for row_number, value in enumerate(cleaned[column].tolist(), start=1):
            if value is None or (isinstance(value, str) and value.strip() == ""):
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains blank {column} at row {row_number}."
                )
            if column in _CATEGORICAL_DOMAINS:
                if not isinstance(value, str) or value not in _CATEGORICAL_DOMAINS[column]:
                    raise FrozenPredictorError(
                        f"RQ2 predictor input contains invalid {column} at row {row_number}."
                    )
                converted.append("Exempt" if value == "1111" else value)
                continue
            if column == "debt_to_income_ratio":
                if not isinstance(value, str) or value not in _DTI_DOMAIN:
                    raise FrozenPredictorError(
                        f"RQ2 predictor input contains invalid DTI at row {row_number}."
                    )
                converted.append(value)
                continue
            if isinstance(value, str) and value in _NUMERIC_SPECIAL[column]:
                converted.append(value)
                continue
            if isinstance(value, (bool, np.bool_)):
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains invalid numeric {column} at row {row_number}."
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains invalid numeric {column} at row {row_number}."
                ) from exc
            if not math.isfinite(number):
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains non-finite {column} at row {row_number}."
                )
            if column in {"loan_amount", "loan_to_value_ratio", "loan_term"} and number <= 0:
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains nonpositive {column} at row {row_number}."
                )
            if column == "loan_term" and not number.is_integer():
                raise FrozenPredictorError(
                    f"RQ2 predictor input contains non-integer loan_term at row {row_number}."
                )
            converted.append(0.0 if column == "income" and number < 0 else number)
        cleaned[column] = pd.Series(converted, index=cleaned.index, dtype=object)
    return cleaned.reset_index(drop=True)


def _validate_probability_vector(values: Any, *, rows: int, model_key: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (rows,):
        raise FrozenPredictorError(
            f"{model_key} must return one p_approved value per row; got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise FrozenPredictorError(f"{model_key} returned a non-finite p_approved value.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise FrozenPredictorError(f"{model_key} returned p_approved outside [0, 1].")
    return probabilities


def _approved_from_two_class_matrix(
    matrix: Any,
    classes: Any,
    *,
    rows: int,
    model_key: str,
) -> np.ndarray:
    probabilities = np.asarray(matrix, dtype=np.float64)
    class_order = np.asarray(classes)
    approved = np.flatnonzero(class_order == 1)
    if probabilities.shape != (rows, 2) or class_order.shape != (2,) or approved.shape != (1,):
        raise FrozenPredictorError(f"{model_key} returned an invalid two-class probability matrix.")
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
        )
    ):
        raise FrozenPredictorError(f"{model_key} returned invalid class probabilities.")
    return _validate_probability_vector(
        probabilities[:, int(approved[0])], rows=rows, model_key=model_key
    )


class _BaseFrozenPredictor:
    def __init__(self, spec: FrozenPredictorSpec) -> None:
        self.model_key = spec.model_key
        self.run_id = spec.run_id
        self.execution_device = "cpu"
        self._spec = spec

    def predict_p_approved(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_p_approved_timed(X).probabilities

    def predict_p_approved_timed(self, X: pd.DataFrame) -> TimedPrediction:
        preprocessing_start = time.perf_counter_ns()
        cleaned = _canonical_feature_frame(X)
        prepared = self._prepare(cleaned)
        preprocessing_end = time.perf_counter_ns()
        values = self._infer(prepared, rows=len(cleaned))
        inference_end = time.perf_counter_ns()
        probabilities = _validate_probability_vector(
            values, rows=len(cleaned), model_key=self.model_key
        )
        return TimedPrediction(
            probabilities=probabilities,
            preprocessing_seconds=(preprocessing_end - preprocessing_start) / 1.0e9,
            inference_seconds=(inference_end - preprocessing_end) / 1.0e9,
        )

    def smoke_compare_stored(
        self, validation_rows: pd.DataFrame, *, atol: float
    ) -> None:
        required = {"record_id", *FEATURE_COLUMNS}
        if not isinstance(validation_rows, pd.DataFrame) or not required.issubset(
            validation_rows.columns
        ):
            raise FrozenPredictorError(
                "Smoke input must contain record_id and the ordered final 15 fields."
            )
        if len(validation_rows) == 0:
            raise FrozenPredictorError("Smoke input must contain at least one Validation row.")
        record_ids = validation_rows["record_id"].astype(str)
        if record_ids.eq("").any() or record_ids.duplicated().any():
            raise FrozenPredictorError("Smoke record_id values must be nonblank and unique.")
        if not isinstance(atol, (int, float)) or isinstance(atol, bool) or atol < 0:
            raise FrozenPredictorError("Smoke comparison tolerance must be nonnegative.")

        stored_path = self._spec.run_directory / "validation" / "validation_predictions.csv"
        try:
            stored = pd.read_csv(
                stored_path,
                usecols=["record_id", "p_approved"],
                dtype={"record_id": str},
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise FrozenPredictorError(
                f"Cannot read frozen Validation predictions for {self.model_key}."
            ) from exc
        if stored["record_id"].isna().any() or stored["record_id"].duplicated().any():
            raise FrozenPredictorError(
                f"Frozen Validation prediction IDs are invalid for {self.model_key}."
            )
        indexed = stored.set_index("record_id")
        missing = [record_id for record_id in record_ids if record_id not in indexed.index]
        if missing:
            raise FrozenPredictorError(
                f"Frozen Validation predictions are missing smoke IDs for {self.model_key}."
            )
        expected = indexed.loc[record_ids.tolist(), "p_approved"].to_numpy(dtype=np.float64)
        observed = self.predict_p_approved(validation_rows.loc[:, FEATURE_COLUMNS])
        if not np.allclose(observed, expected, rtol=0.0, atol=float(atol)):
            maximum_error = float(np.max(np.abs(observed - expected)))
            raise FrozenPredictorError(
                f"Fresh smoke predictions differ from frozen stored predictions for "
                f"{self.model_key}; max_abs_error={maximum_error}."
            )

    def _prepare(self, X: pd.DataFrame) -> Any:
        return X

    def _infer(self, prepared: Any, *, rows: int) -> np.ndarray:
        raise NotImplementedError


class _SklearnPipelinePredictor(_BaseFrozenPredictor):
    def __init__(self, spec: FrozenPredictorSpec, pipeline: Any) -> None:
        super().__init__(spec)
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise FrozenPredictorError(f"{spec.model_key} bundle has no predict_proba method.")
        self._pipeline = pipeline
        self._split_pipeline = hasattr(pipeline, "steps") and len(pipeline.steps) >= 2
        self._estimator = pipeline.steps[-1][1] if self._split_pipeline else pipeline

    def _prepare(self, X: pd.DataFrame) -> Any:
        return self._pipeline[:-1].transform(X) if self._split_pipeline else X

    def _infer(self, prepared: Any, *, rows: int) -> np.ndarray:
        return _approved_from_two_class_matrix(
            self._estimator.predict_proba(prepared),
            getattr(self._estimator, "classes_", None),
            rows=rows,
            model_key=self.model_key,
        )


class _PlainMLPPredictor(_BaseFrozenPredictor):
    def __init__(self, spec: FrozenPredictorSpec, preprocessing: Any, estimator: Any) -> None:
        super().__init__(spec)
        self._preprocessing = preprocessing
        self._estimator = estimator

    def _prepare(self, X: pd.DataFrame) -> Any:
        return self._preprocessing.transform(X)

    def _infer(self, prepared: Any, *, rows: int) -> np.ndarray:
        return _approved_from_two_class_matrix(
            self._estimator.predict_proba(prepared),
            getattr(self._estimator, "classes_", None),
            rows=rows,
            model_key=self.model_key,
        )


class _RealMLPPredictor(_BaseFrozenPredictor):
    def __init__(self, spec: FrozenPredictorSpec, preprocessing: Any, estimator: Any) -> None:
        super().__init__(spec)
        self._preprocessing = preprocessing
        self._estimator = estimator

    def _prepare(self, X: pd.DataFrame) -> Any:
        return self._preprocessing.transform(X)

    def _infer(self, prepared: Any, *, rows: int) -> np.ndarray:
        return _approved_from_two_class_matrix(
            self._estimator.predict_proba(prepared),
            getattr(self._estimator, "classes_", None),
            rows=rows,
            model_key=self.model_key,
        )


class _TabMMiniPredictor(_BaseFrozenPredictor):
    def __init__(
        self,
        spec: FrozenPredictorSpec,
        preprocessing: Any,
        model: Any,
        *,
        device: str,
        batch_size: int,
    ) -> None:
        super().__init__(spec)
        if batch_size <= 0:
            raise FrozenPredictorError("TabMmini inference batch size must be positive.")
        self._preprocessing = preprocessing
        self._model = model
        self.device = device
        self.batch_size = batch_size

    def _prepare(self, X: pd.DataFrame) -> Any:
        return self._preprocessing.transform(X)

    def _infer(self, prepared: Any, *, rows: int) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, rows, self.batch_size):
                stop = min(start + self.batch_size, rows)
                values = self._model.predict_approved_probability(
                    prepared.x_num[start:stop].to(self.device),
                    prepared.x_cat[start:stop].to(self.device),
                )
                chunks.append(values.detach().cpu().numpy())
        return np.concatenate(chunks).astype(np.float64, copy=False)


def _load_plain_mlp_components(path: Path, expected_run_id: str) -> tuple[Any, Any]:
    try:
        from src.dataset_b.gsv_sl.cpu_inference import load_plain_mlp_cpu

        preprocessing, estimator = load_plain_mlp_cpu(
            path, expected_run_id=expected_run_id
        )
    except Exception as exc:
        raise FrozenPredictorError("Cannot load the frozen PlainMLP bundle.") from exc
    return preprocessing, estimator


def _load_realmlp_components(path: Path) -> tuple[Any, Any]:
    try:
        from src.dataset_b.gsv_sl.cpu_inference import CPUInferenceError, load_realmlp_cpu

        return load_realmlp_cpu(path)
    except CPUInferenceError as exc:
        raise FrozenPredictorError(str(exc)) from exc
    except Exception as exc:
        raise FrozenPredictorError("Cannot load the frozen RealMLP bundle.") from exc


def _load_tabmmini_components(
    spec: FrozenPredictorSpec, device: str | None
) -> tuple[Any, Any, str, int]:
    try:
        import torch

        from src.dataset_b.models.tabmmini import TabMMiniModel
    except Exception as exc:
        raise FrozenPredictorError("Cannot import the TabMmini inference runtime.") from exc

    selected_device = device or "cpu"
    if selected_device != "cpu":
        raise FrozenPredictorError("RQ2 TabMmini inference must run on CPU.")
    preprocessor_path = spec.run_directory / "full_train" / "model" / "preprocessor.joblib"
    if not preprocessor_path.is_file() or preprocessor_path.stat().st_size == 0:
        raise FrozenPredictorError("Frozen TabMmini preprocessor is missing or empty.")
    try:
        preprocessing = joblib.load(preprocessor_path)
        model, payload = TabMMiniModel.load_inference_bundle(
            spec.bundle_path, map_location=selected_device
        )
        model.to(selected_device)
    except Exception as exc:
        raise FrozenPredictorError("Cannot load the frozen TabMmini inference bundle.") from exc
    if payload.get("run_id") != spec.run_id:
        raise FrozenPredictorError("Frozen TabMmini bundle run_id differs from the protocol.")
    batch_size = payload.get("effective_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FrozenPredictorError("Frozen TabMmini bundle has an invalid inference batch size.")
    return preprocessing, model, selected_device, batch_size


def load_frozen_predictor(
    model_key: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    tabmmini_device: str | None = None,
) -> FrozenPredictor:
    """Load one protocol-listed frozen predictor after read-only preflight checks."""

    spec = _predictor_spec(model_key, artifact_root)
    _preflight(spec)
    if model_key in {"lr", "rf", "xgboost"}:
        try:
            pipeline = joblib.load(spec.bundle_path)
        except Exception as exc:
            raise FrozenPredictorError(f"Cannot load the frozen {model_key} pipeline.") from exc
        return _SklearnPipelinePredictor(spec, pipeline)
    if model_key == "plain_mlp":
        preprocessing, estimator = _load_plain_mlp_components(spec.bundle_path, spec.run_id)
        return _PlainMLPPredictor(spec, preprocessing, estimator)
    if model_key == "realmlp":
        preprocessing, estimator = _load_realmlp_components(spec.bundle_path)
        return _RealMLPPredictor(spec, preprocessing, estimator)

    preprocessing, model, device, batch_size = _load_tabmmini_components(
        spec, tabmmini_device
    )
    return _TabMMiniPredictor(
        spec,
        preprocessing,
        model,
        device=device,
        batch_size=batch_size,
    )
