"""Shared final-run contract for the Dataset B RealMLP 15-feature pipeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.dataset_b.preprocessing import (
    DatasetBInputQualityGate,
    DatasetBSplitLoader,
    make_realmlp_native_preprocessing_pipeline,
)
from src.dataset_b.preprocessing.categorical_encoder import (
    CATEGORICAL_COLUMNS,
    NUMERIC_VALUE_COLUMNS,
    TWO_CHANNEL_COLUMNS,
)
from src.utils.path import PROJECT_ROOT


RUN_ID_PATTERN = re.compile(r"^realmlp_final_15f_\d{8}_\d{2}$")
FEATURE_VERSION = "feature_set_v1"
SPLIT_VERSION = "split_v1"
PREPROCESSING_VERSION = "dataset_b_neural_preprocessing_v1"
RAW_FEATURE_COUNT = 15
SEMANTIC_FEATURE_COUNT = 19
EXPECTED_ROWS = {"train": 205_310, "validation": 43_995, "test": 43_996}


class RealMLPFinalContractError(RuntimeError):
    """Raised when final RealMLP source or artifacts violate the locked contract."""


@dataclass(frozen=True)
class RealMLPFinalPaths:
    """Run-scoped paths; historical ``realmlp_v1`` paths are intentionally excluded."""

    run_id: str
    working_root: Path
    final_root: Path

    @classmethod
    def for_run(
        cls,
        run_id: str,
        *,
        artifact_root: str | Path = PROJECT_ROOT / "artifacts" / "dataset_b" / "realmlp",
    ) -> "RealMLPFinalPaths":
        validate_run_id(run_id)
        root = Path(artifact_root)
        return cls(
            run_id=run_id,
            working_root=root / "working" / run_id,
            final_root=root / "final" / run_id,
        )

    def stage(self, name: str, *, final: bool = False) -> Path:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise RealMLPFinalContractError("RealMLP stage name is invalid.")
        return (self.final_root if final else self.working_root) / name


@dataclass(frozen=True)
class PreparedNativePartition:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    record_ids: tuple[str, ...]
    quality_audit: dict[str, Any]


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RealMLPFinalContractError(
            "run_id must match realmlp_final_15f_<YYYYMMDD>_<NN>."
        )
    return run_id


def load_quality_gated_partition(
    partition: Literal["train", "validation", "test"],
    *,
    loader: DatasetBSplitLoader | None = None,
) -> Any:
    """Load one immutable partition and apply the same central gate as PlainMLP."""
    active_loader = loader or DatasetBSplitLoader()
    source = active_loader.load(partition)
    gated = DatasetBInputQualityGate().apply(source)
    expected = EXPECTED_ROWS[partition]
    if len(gated.X) != expected or len(gated.record_ids) != expected:
        raise RealMLPFinalContractError(
            f"RealMLP {partition} row count differs from split_v1: {len(gated.X)}."
        )
    if tuple(gated.X.columns) != tuple(active_loader.feature_columns):
        raise RealMLPFinalContractError("Quality-gated RealMLP raw schema is not the final 15 fields.")
    return gated


def fit_native_preprocessing(gated: Any) -> tuple[Any, PreparedNativePartition]:
    pipeline = make_realmlp_native_preprocessing_pipeline()
    native = pipeline.fit_transform(gated.X.reset_index(drop=True))
    prepared = _prepared(gated, native)
    return pipeline, prepared


def transform_native_preprocessing(pipeline: Any, gated: Any) -> PreparedNativePartition:
    native = pipeline.transform(gated.X.reset_index(drop=True))
    return _prepared(gated, native)


def _prepared(gated: Any, native: pd.DataFrame) -> PreparedNativePartition:
    if tuple(native.columns) != TWO_CHANNEL_COLUMNS or len(native.columns) != SEMANTIC_FEATURE_COUNT:
        raise RealMLPFinalContractError("RealMLP native frame is not the ordered 19-column schema.")
    if any(native[column].dtype != np.dtype("float32") for column in NUMERIC_VALUE_COLUMNS):
        raise RealMLPFinalContractError("RealMLP native numerical columns are not float32.")
    if any(not isinstance(native[column].dtype, pd.CategoricalDtype) for column in CATEGORICAL_COLUMNS):
        raise RealMLPFinalContractError("RealMLP native categorical columns are not category dtype.")
    ids = tuple(gated.record_ids.astype(str).tolist())
    if len(ids) != len(set(ids)) or len(ids) != len(native):
        raise RealMLPFinalContractError("RealMLP native rows do not align with unique record_id values.")
    return PreparedNativePartition(
        name=gated.name,
        X=native,
        y=gated.y.to_numpy(dtype=np.int8),
        record_ids=ids,
        quality_audit=gated.quality_audit.as_dict(),
    )


def validate_probabilities(probabilities: Any, rows: int) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if (
        matrix.shape != (rows, 2)
        or not np.isfinite(matrix).all()
        or np.any((matrix < 0.0) | (matrix > 1.0))
        or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise RealMLPFinalContractError("RealMLP probability matrix is invalid.")
    return matrix


def base_manifest(run_id: str, stage: str, *, status: str = "PASS") -> dict[str, Any]:
    validate_run_id(run_id)
    return {
        "schema_version": 1,
        "dataset": "dataset_b",
        "model": "realmlp",
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_feature_count": RAW_FEATURE_COUNT,
        "semantic_feature_count": SEMANTIC_FEATURE_COUNT,
    }


def validate_manifest(payload: dict[str, Any], *, run_id: str, stage: str) -> dict[str, Any]:
    expected = base_manifest(run_id, stage)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RealMLPFinalContractError(f"RealMLP manifest differs for {key}.")
    return payload


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMLPFinalContractError(f"Cannot read RealMLP JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RealMLPFinalContractError(f"RealMLP JSON must contain an object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def begin_stage(paths: RealMLPFinalPaths, stage: str) -> tuple[Path, Path]:
    destination = paths.stage(stage)
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise RealMLPFinalContractError(
            f"RealMLP final stage already exists and will not be overwritten: {destination}"
        )
    paths.working_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True)
    return temporary, destination


def publish_stage(temporary: Path, destination: Path) -> None:
    if destination.exists() or not temporary.is_dir():
        raise RealMLPFinalContractError("RealMLP stage cannot be published atomically.")
    os.replace(temporary, destination)
