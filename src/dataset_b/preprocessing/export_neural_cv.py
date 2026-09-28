"""Export atomic five-fold neural preprocessing artifacts from Dataset B Train."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from src.dataset_b.splits import DatasetBSplitter
from src.utils.path import CONFIG_DIR, PROJECT_ROOT

from .categorical_encoder import (
    CATEGORICAL_COLUMNS, CATEGORICAL_SCHEMA, ENCODED_FEATURE_COUNT,
    STATUS_REFERENCE_CATEGORIES,
)
from .neural_profiles import make_plain_mlp_preprocessing_pipeline
from .raw_parser import DatasetBRawParser
from .split_inputs import DatasetBSplitLoader


_LOGGER = logging.getLogger(__name__)
PREPROCESSING_VERSION = "dataset_b_neural_preprocessing_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "preprocessing" / "dataset_b_v1" / "cv_neural_preprocessing_v1"
)
_SOURCE_FILES = {
    "feature_contract": CONFIG_DIR / "dataset_b" / "feature_set_v1.yaml",
    "split_contract": CONFIG_DIR / "dataset_b" / "split_v1.yaml",
    "plain_mlp_config": CONFIG_DIR / "dataset_b" / "plain_mlp_v1.yaml",
    "split_manifest": PROJECT_ROOT / "artifacts" / "dataset_b" / "split_v1.json",
    "neural_requirements": PROJECT_ROOT / "requirements.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _categorical_frame(parsed: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    numeric_status_columns = dict(STATUS_REFERENCE_CATEGORIES)
    for column in CATEGORICAL_COLUMNS:
        if column in numeric_status_columns:
            source = column.removesuffix("_status")
            values[column] = parsed[source].map(
                lambda value: value if isinstance(value, str) else "OBSERVED"
            )
        else:
            values[column] = parsed[column]
    return pd.DataFrame(values, index=parsed.index)


def _preflight_category_support(
    X: pd.DataFrame,
    fold_positions: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    parsed = DatasetBRawParser().parse(X).cleaned
    categories = _categorical_frame(parsed)
    fold_support: list[dict[str, object]] = []
    failures: list[str] = []
    for fold_number, (train_positions, validation_positions) in enumerate(fold_positions, start=1):
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
                        "Neural category absent from fold-train and fold-validation fold=%d column=%s category=%s",
                        fold_number, column, category,
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
            "Dataset B neural absent-category gate failed before artifact creation: "
            + "; ".join(failures)
        )
    return fold_support


def _validate_matrix(matrix: csr_matrix, rows: int) -> None:
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Dataset B neural preprocessing must produce a CSR matrix.")
    if matrix.shape != (rows, ENCODED_FEATURE_COUNT):
        raise ValueError("Dataset B neural preprocessing produced an unexpected matrix shape.")
    if matrix.dtype != np.float32:
        raise ValueError("Dataset B neural preprocessing must produce float32 matrices.")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Dataset B neural preprocessing produced NaN or infinity.")


def export_neural_cv_preprocessed(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Export five fold-local plain-MLP matrices without reading Validation/Test."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Dataset B neural preprocessing export already exists: {destination}")

    loader = DatasetBSplitLoader()
    train = loader.load("train")
    splitter = DatasetBSplitter.from_yaml().cv_splitter()
    fold_positions = list(splitter.split(train.X, train.y))
    if len(fold_positions) != 5:
        raise ValueError("Dataset B neural preprocessing requires exactly five Train folds.")

    category_support = _preflight_category_support(train.X, fold_positions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        folds: list[dict[str, object]] = []
        artifact_hashes: dict[str, str] = {}
        expected_names: list[str] | None = None
        expected_sources: list[str] | None = None
        expected_categories: list[str | None] | None = None

        for fold_number, (train_positions, validation_positions) in enumerate(fold_positions, start=1):
            fold_dir = staging / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            X_train = train.X.iloc[train_positions]
            X_validation = train.X.iloc[validation_positions]
            profile = make_plain_mlp_preprocessing_pipeline()
            train_matrix = profile.fit_transform(X_train, train.y.iloc[train_positions])
            train_log_audit = dict(profile.named_steps["log1p_numeric"].last_audit_)
            validation_matrix = profile.transform(X_validation)
            validation_log_audit = dict(profile.named_steps["log1p_numeric"].last_audit_)
            _validate_matrix(train_matrix, len(train_positions))
            _validate_matrix(validation_matrix, len(validation_positions))

            encoder = profile.named_steps["encode"]
            names = profile.get_feature_names_out().tolist()
            sources = list(encoder.feature_sources_)
            categories = list(encoder.feature_categories_)
            if expected_names is None:
                expected_names, expected_sources, expected_categories = names, sources, categories
            elif (names, sources, categories) != (expected_names, expected_sources, expected_categories):
                raise ValueError("Dataset B neural encoded column contract changed between folds.")

            train_path = fold_dir / "plain_mlp_train.npz"
            validation_path = fold_dir / "plain_mlp_fold_validation.npz"
            rows_path = fold_dir / "rows.npz"
            save_npz(train_path, train_matrix)
            save_npz(validation_path, validation_matrix)
            np.savez_compressed(
                rows_path,
                train_positions=train_positions,
                fold_validation_positions=validation_positions,
                train_record_ids=train.record_ids.iloc[train_positions].to_numpy(dtype=str),
                fold_validation_record_ids=train.record_ids.iloc[validation_positions].to_numpy(dtype=str),
                train_y=train.y.iloc[train_positions].to_numpy(dtype=np.int8),
                fold_validation_y=train.y.iloc[validation_positions].to_numpy(dtype=np.int8),
            )
            for path in (train_path, validation_path, rows_path):
                artifact_hashes[path.relative_to(staging).as_posix()] = _sha256(path)

            folds.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train_positions),
                    "fold_validation_rows": len(validation_positions),
                    "directory": fold_dir.name,
                    "numeric_medians": dict(profile.named_steps["two_channel"].imputer_.medians_),
                    "log_audit": {
                        "fold_train": train_log_audit,
                        "fold_validation": validation_log_audit,
                    },
                    "category_support": category_support[fold_number - 1]["columns"],
                }
            )

        source_hashes = {
            name: {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": _sha256(path)}
            for name, path in _SOURCE_FILES.items()
        }
        manifest = {
            "dataset": "dataset_b",
            "artifact": "fold_local_neural_preprocessing",
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_route": "plain_mlp_external_fixed_schema_ohe",
            "split_version": "split_v1",
            "feature_version": loader.feature_version,
            "source_partition": "train",
            "fit_scope": "fold_train_only",
            "use": "cross_validation_only",
            "fold_count": 5,
            "matrix_count": 10,
            "encoded_feature_count": len(expected_names or ()),
            "matrix_format": "csr",
            "matrix_dtype": "float32",
            "model_boundary": "dense_float32_per_batch",
            "numeric_transform": "fold_train_median_then_log1p",
            "numeric_scaling": "none",
            "status_reference_categories": dict(STATUS_REFERENCE_CATEGORIES),
            "sample_weights_exported": False,
            "library_versions": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "scipy": version("scipy"),
                "torch": version("torch"),
            },
            "output_columns": expected_names,
            "output_raw_sources": expected_sources,
            "output_categories": expected_categories,
            "source_hashes": source_hashes,
            "artifact_hashes": artifact_hashes,
            "folds": folds,
        }
        if manifest["encoded_feature_count"] != ENCODED_FEATURE_COUNT:
            raise ValueError("Dataset B neural schema did not derive the expected 66-column width.")
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(destination)
        final_manifest = destination / "manifest.json"
        _LOGGER.info("Completed Dataset B neural CV preprocessing path=%s", final_manifest)
        return final_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    print(export_neural_cv_preprocessed())
