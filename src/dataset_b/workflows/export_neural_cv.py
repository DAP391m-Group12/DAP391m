"""Canonical Export run-scoped fold-local PlainMLP preprocessing from Dataset B Train."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr, save_npz

from src.dataset_b.data import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    ENCODED_FEATURE_COUNT,
    EXEMPTION_FLAG_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_VALUE_COLUMNS,
    STATUS_REFERENCE_CATEGORIES,
    TWO_CHANNEL_COLUMNS,
    DatasetBInputQualityGate,
    DatasetBSplitLoader,
    DatasetBSplitter,
)
from src.dataset_b.models.classical import LOG1P_VALUE_COLUMNS
from src.dataset_b.models.neural.plain_mlp.preprocessing import make_plain_mlp_preprocessing_pipeline
from src.utils.path import PROJECT_ROOT


_LOGGER = logging.getLogger(__name__)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PREPROCESSING_VERSION = "dataset_b_neural_preprocessing_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "preprocessing" / "dataset_b_v1" / "cv_neural_preprocessing_v1"
)


def _validated_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Dataset B PlainMLP preprocessing run_id must be 1-128 characters "
            "using letters, numbers, dot, underscore or hyphen, and must start "
            "with a letter or number."
        )
    return run_id


def _categorical_frame(normalized: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    numeric_status_columns = dict(STATUS_REFERENCE_CATEGORIES)
    for column in CATEGORICAL_COLUMNS:
        if column in numeric_status_columns:
            source = column.removesuffix("_status")
            values[column] = normalized[source].map(
                lambda value: value if isinstance(value, str) else "OBSERVED"
            )
        else:
            values[column] = normalized[column]
    return pd.DataFrame(values, index=normalized.index)


def _preflight_category_support(
    normalized: pd.DataFrame,
    fold_positions: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    categories = _categorical_frame(normalized)
    fold_support: list[dict[str, object]] = []
    failures: list[str] = []
    for fold_number, (train_positions, validation_positions) in enumerate(
        fold_positions, start=1
    ):
        by_column: dict[str, dict[str, object]] = {}
        for column in CATEGORICAL_COLUMNS:
            support: dict[str, object] = {}
            train_values = categories.iloc[train_positions][column]
            validation_values = categories.iloc[validation_positions][column]
            for category in CATEGORICAL_SCHEMA[column]:
                train_count = int(train_values.eq(category).sum())
                validation_count = int(validation_values.eq(category).sum())
                if train_count == 0 and validation_count > 0:
                    outcome = "fail_absent_train_present_validation"
                    failures.append(
                        f"fold={fold_number} column={column} category={category!r} "
                        f"fold_validation_support={validation_count}"
                    )
                elif train_count == 0:
                    outcome = "warning_absent_both"
                    _LOGGER.warning(
                        "PlainMLP category absent from fold-train and fold-validation "
                        "fold=%d column=%s category=%s",
                        fold_number,
                        column,
                        category,
                    )
                else:
                    outcome = "pass_supported_in_fold_train"
                support[category] = {
                    "fold_train": train_count,
                    "fold_validation": validation_count,
                    "outcome": outcome,
                }
            by_column[column] = support
        fold_support.append({"fold": fold_number, "columns": by_column})
    if failures:
        raise ValueError(
            "Dataset B PlainMLP absent-category gate failed before artifact creation: "
            + "; ".join(failures)
        )
    return fold_support


def _validate_matrix(matrix: object, expected_rows: int) -> None:
    if not isspmatrix_csr(matrix):
        raise ValueError("Dataset B PlainMLP preprocessing must produce SciPy CSR.")
    if matrix.shape != (expected_rows, ENCODED_FEATURE_COUNT):
        raise ValueError("Dataset B PlainMLP preprocessing produced an unexpected shape.")
    if matrix.dtype != np.dtype(np.float64):
        raise ValueError("Dataset B PlainMLP preprocessing must produce float64 matrices.")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Dataset B PlainMLP preprocessing produced NaN or infinity.")


def export_neural_cv_preprocessed(
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Export five PlainMLP CV folds without reading Validation or Test.

    ``run_id`` is mandatory. The central quality gate runs exactly once before
    any fitted preprocessing. The destination must not already exist.
    """
    run_id = _validated_run_id(run_id)
    root = Path(output_root)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(
            f"Dataset B PlainMLP preprocessing run already exists: {destination}"
        )

    loader = DatasetBSplitLoader()
    raw_train = loader.load("train")
    train = DatasetBInputQualityGate().apply(raw_train)
    if tuple(train.X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Dataset B quality-gated Train must contain the exact normalized "
            "15-field schema."
        )
    if train.quality_audit.rejected_rows != 0:
        raise ValueError("Dataset B PlainMLP quality gate rejected one or more records.")

    splitter = DatasetBSplitter.from_yaml().cv_splitter()
    fold_positions = [
        (
            np.asarray(train_positions, dtype=np.int64),
            np.asarray(validation_positions, dtype=np.int64),
        )
        for train_positions, validation_positions in splitter.split(train.X, train.y)
    ]
    if len(fold_positions) != 5:
        raise ValueError("Dataset B PlainMLP preprocessing requires exactly five Train folds.")

    category_support = _preflight_category_support(train.X, fold_positions)
    root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        folds: list[dict[str, object]] = []
        expected_names: list[str] | None = None
        expected_sources: list[str] | None = None
        expected_categories: list[str | None] | None = None
        expected_semantic_columns: list[str] | None = None
        validation_coverage = np.zeros(len(train.X), dtype=np.int8)

        for fold_number, (train_positions, validation_positions) in enumerate(
            fold_positions, start=1
        ):
            if np.intersect1d(train_positions, validation_positions).size:
                raise ValueError("Dataset B PlainMLP fold train and validation overlap.")
            if len(train_positions) + len(validation_positions) != len(train.X):
                raise ValueError("Dataset B PlainMLP fold does not partition Train.")
            combined = np.concatenate((train_positions, validation_positions))
            if not np.array_equal(
                np.sort(combined), np.arange(len(train.X), dtype=np.int64)
            ):
                raise ValueError(
                    "Dataset B PlainMLP fold positions do not exactly partition Train."
                )
            validation_coverage[validation_positions] += 1

            fold_dir = working / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            X_train = train.X.iloc[train_positions]
            X_validation = train.X.iloc[validation_positions]
            y_train = train.y.iloc[train_positions]
            profile = make_plain_mlp_preprocessing_pipeline()
            train_matrix = profile.fit_transform(X_train, y_train)
            train_log_audit = dict(profile.named_steps["log1p_numeric"].last_audit_)
            validation_matrix = profile.transform(X_validation)
            validation_log_audit = dict(
                profile.named_steps["log1p_numeric"].last_audit_
            )
            _validate_matrix(train_matrix, len(train_positions))
            _validate_matrix(validation_matrix, len(validation_positions))

            names = profile.get_feature_names_out().tolist()
            semantic_columns = (
                profile.named_steps["two_channel"].get_feature_names_out().tolist()
            )
            encoder = profile.named_steps["encode"]
            sources = list(encoder.feature_sources_)
            categories = list(encoder.feature_categories_)
            current_schema = (names, sources, categories, semantic_columns)
            expected_schema = (
                expected_names,
                expected_sources,
                expected_categories,
                expected_semantic_columns,
            )
            if expected_names is None:
                (
                    expected_names,
                    expected_sources,
                    expected_categories,
                    expected_semantic_columns,
                ) = current_schema
            elif current_schema != expected_schema:
                raise ValueError(
                    "Dataset B PlainMLP encoded or semantic schema changed between folds."
                )

            train_ids = train.record_ids.iloc[train_positions].to_numpy(dtype=str)
            validation_ids = train.record_ids.iloc[validation_positions].to_numpy(
                dtype=str
            )
            train_y = y_train.to_numpy(dtype=np.int8)
            validation_y = train.y.iloc[validation_positions].to_numpy(dtype=np.int8)
            if not (
                len(train_ids) == len(train_y) == len(train_positions)
                and len(validation_ids)
                == len(validation_y)
                == len(validation_positions)
            ):
                raise ValueError(
                    "Dataset B PlainMLP row identifiers, labels and positions are misaligned."
                )

            train_path = fold_dir / "plain_mlp_train.npz"
            validation_path = fold_dir / "plain_mlp_fold_validation.npz"
            rows_path = fold_dir / "rows.npz"
            save_npz(train_path, train_matrix)
            save_npz(validation_path, validation_matrix)
            np.savez_compressed(
                rows_path,
                train_positions=train_positions,
                fold_validation_positions=validation_positions,
                train_record_ids=train_ids,
                fold_validation_record_ids=validation_ids,
                train_y=train_y,
                fold_validation_y=validation_y,
            )

            scaler = profile.named_steps["scale_numeric"].scaler_
            folds.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train_positions),
                    "fold_validation_rows": len(validation_positions),
                    "directory": fold_dir.name,
                    "numeric_medians": dict(
                        profile.named_steps["two_channel"].imputer_.medians_
                    ),
                    "numeric_observed_counts": dict(
                        profile.named_steps["two_channel"].imputer_.observed_counts_
                    ),
                    "selective_log1p_columns": list(LOG1P_VALUE_COLUMNS),
                    "unchanged_numeric_columns": [
                        column
                        for column in NUMERIC_VALUE_COLUMNS
                        if column not in LOG1P_VALUE_COLUMNS
                    ],
                    "robust_scaled_columns": list(NUMERIC_VALUE_COLUMNS),
                    "robust_center": scaler.center_.tolist(),
                    "robust_scale": scaler.scale_.tolist(),
                    "log_audit": {
                        "fold_train": train_log_audit,
                        "fold_validation": validation_log_audit,
                    },
                    "category_support": category_support[fold_number - 1]["columns"],
                }
            )

        if len(folds) != 5 or not np.all(validation_coverage == 1):
            raise ValueError(
                "Dataset B PlainMLP fold-validation sets must cover every Train "
                "row exactly once."
            )
        if expected_semantic_columns != list(TWO_CHANNEL_COLUMNS):
            raise ValueError(
                "Dataset B PlainMLP semantic schema must contain the exact 19 columns."
            )
        if len(expected_names or ()) != ENCODED_FEATURE_COUNT:
            raise ValueError(
                "Dataset B PlainMLP encoded schema must contain exactly 62 columns."
            )

        quality = train.quality_audit.as_dict()
        manifest = {
            "dataset": "dataset_b",
            "artifact": "fold_local_neural_preprocessing",
            "run_id": run_id,
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_route": "plain_mlp_external_fixed_schema_ohe",
            "split_version": "split_v1",
            "feature_version": loader.feature_version,
            "source_partition": "train",
            "fit_scope": "fold_train_only",
            "use": "cross_validation_only",
            "fold_count": 5,
            "matrix_count": 10,
            "raw_feature_count": len(FEATURE_COLUMNS),
            "semantic_feature_count": len(TWO_CHANNEL_COLUMNS),
            "encoded_feature_count": ENCODED_FEATURE_COUNT,
            "raw_feature_columns": list(FEATURE_COLUMNS),
            "semantic_columns": expected_semantic_columns,
            "matrix_contract": {
                "format": "csr",
                "dtype": "float64",
                "finite": True,
                "full_matrix_densification": False,
            },
            "model_boundary": "dense_float64_per_batch",
            "numeric_transforms": {
                "median_imputation": "fold_train_only",
                "log1p_columns": list(LOG1P_VALUE_COLUMNS),
                "robust_scaled_columns": list(NUMERIC_VALUE_COLUMNS),
                "ohe_scaled": False,
            },
            "status_reference_categories": dict(STATUS_REFERENCE_CATEGORIES),
            "categorical_normalization": {
                "raw_token": "1111",
                "semantic_token": "Exempt",
                "columns": list(EXEMPTION_FLAG_COLUMNS),
                "missing_category_retained": False,
            },
            "quality_gate": quality,
            "sample_weights_exported": False,
            "library_versions": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "scipy": version("scipy"),
            },
            "output_columns": expected_names,
            "output_raw_sources": expected_sources,
            "output_categories": expected_categories,
            "fold_validation_coverage": {
                "rows": len(validation_coverage),
                "minimum_assignments": int(validation_coverage.min()),
                "maximum_assignments": int(validation_coverage.max()),
                "exactly_once": True,
            },
            "folds": folds,
        }
        manifest_path = working / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError(
                f"Dataset B PlainMLP preprocessing run already exists: {destination}"
            )
        working.rename(destination)
        final_manifest = destination / "manifest.json"
        _LOGGER.info(
            "Completed Dataset B PlainMLP CV preprocessing run_id=%s folds=%d path=%s",
            run_id,
            len(folds),
            final_manifest,
        )
        return final_manifest
    except Exception:
        if working.exists():
            shutil.rmtree(working)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(export_neural_cv_preprocessed(args.run_id, args.output_root))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _main()
