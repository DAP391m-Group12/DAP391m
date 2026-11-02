"""Export run-scoped fold-local Dataset B preprocessing for classical CV."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import isspmatrix_csr, save_npz

from src.dataset_b.data import (
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
from src.dataset_b.models.classical import LOG1P_VALUE_COLUMNS, make_preprocessing_pipeline
from src.utils.path import PROJECT_ROOT


_LOGGER = logging.getLogger(__name__)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PREPROCESSING_VERSION = "dataset_b_preprocessing_v2"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "preprocessing" / "dataset_b_v1" / "cv_preprocessing_v2"
)


def _validated_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Dataset B preprocessing run_id must be 1-128 characters using letters, "
            "numbers, dot, underscore or hyphen, and must start with a letter or number."
        )
    return run_id


def _validate_matrix(matrix: object, expected_rows: int, profile: str) -> None:
    if not isspmatrix_csr(matrix):
        raise ValueError(f"Dataset B {profile} matrix must be SciPy CSR.")
    if matrix.shape != (expected_rows, ENCODED_FEATURE_COUNT):
        raise ValueError(f"Dataset B {profile} matrix has an unexpected shape.")
    if matrix.dtype != np.dtype(np.float64):
        raise ValueError(f"Dataset B {profile} matrix must use float64.")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Dataset B {profile} matrix contains non-finite values.")


def export_cv_preprocessed(
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write five fold-specific LR and shared RF/XGBoost matrices.

    ``run_id`` is mandatory and the destination must not already exist. Only
    the existing Train partition is loaded. The input-quality gate runs before
    CV; every profile is then fitted afresh on its own fold-train rows.
    """
    run_id = _validated_run_id(run_id)
    root = Path(output_root)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(f"Dataset B preprocessing run already exists: {destination}")

    loader = DatasetBSplitLoader()
    raw_train = loader.load("train")
    train = DatasetBInputQualityGate().apply(raw_train)
    if tuple(train.X.columns) != FEATURE_COLUMNS:
        raise ValueError("Dataset B quality-gated Train must contain the exact 15-field schema.")

    splitter = DatasetBSplitter.from_yaml().cv_splitter()
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
            splitter.split(train.X, train.y), start=1
        ):
            train_positions = np.asarray(train_positions, dtype=np.int64)
            validation_positions = np.asarray(validation_positions, dtype=np.int64)
            if np.intersect1d(train_positions, validation_positions).size:
                raise ValueError("Dataset B CV fold train and validation positions overlap.")
            if len(train_positions) + len(validation_positions) != len(train.X):
                raise ValueError("Dataset B CV fold does not partition retained Train rows.")
            combined_positions = np.concatenate((train_positions, validation_positions))
            if not np.array_equal(
                np.sort(combined_positions),
                np.arange(len(train.X), dtype=np.int64),
            ):
                raise ValueError("Dataset B CV fold positions do not exactly partition Train.")
            validation_coverage[validation_positions] += 1

            fold_dir = working / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            X_train = train.X.iloc[train_positions]
            X_validation = train.X.iloc[validation_positions]
            y_train = train.y.iloc[train_positions]
            profile_stats: dict[str, object] = {}

            for export_name, profile_name in (("lr", "lr"), ("tree", "rf")):
                profile = make_preprocessing_pipeline(profile_name)
                train_matrix = profile.fit_transform(X_train, y_train)
                validation_matrix = profile.transform(X_validation)
                _validate_matrix(train_matrix, len(train_positions), export_name)
                _validate_matrix(validation_matrix, len(validation_positions), export_name)

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
                        "Dataset B CV encoded or semantic schema changed between folds or profiles."
                    )

                save_npz(fold_dir / f"{export_name}_train.npz", train_matrix)
                save_npz(
                    fold_dir / f"{export_name}_fold_validation.npz",
                    validation_matrix,
                )

                stats: dict[str, object] = {
                    "numeric_medians": dict(
                        profile.named_steps["two_channel"].imputer_.medians_
                    ),
                    "numeric_observed_counts": dict(
                        profile.named_steps["two_channel"].imputer_.observed_counts_
                    ),
                }
                if profile_name == "lr":
                    scaler = profile.named_steps["scale_numeric"].scaler_
                    stats.update(
                        {
                            "selective_log1p_columns": list(LOG1P_VALUE_COLUMNS),
                            "robust_scaled_columns": list(NUMERIC_VALUE_COLUMNS),
                            "robust_center": scaler.center_.tolist(),
                            "robust_scale": scaler.scale_.tolist(),
                        }
                    )
                else:
                    stats.update(
                        {
                            "selective_log1p_columns": [],
                            "robust_scaled_columns": [],
                        }
                    )
                profile_stats[export_name] = stats

            train_ids = train.record_ids.iloc[train_positions].to_numpy(dtype=str)
            validation_ids = train.record_ids.iloc[validation_positions].to_numpy(dtype=str)
            train_y = train.y.iloc[train_positions].to_numpy(dtype=np.int8)
            validation_y = train.y.iloc[validation_positions].to_numpy(dtype=np.int8)
            if not (
                len(train_ids) == len(train_y) == len(train_positions)
                and len(validation_ids) == len(validation_y) == len(validation_positions)
            ):
                raise ValueError("Dataset B CV row identifiers, labels and positions are misaligned.")

            np.savez_compressed(
                fold_dir / "rows.npz",
                train_positions=train_positions,
                fold_validation_positions=validation_positions,
                train_record_ids=train_ids,
                fold_validation_record_ids=validation_ids,
                train_y=train_y,
                fold_validation_y=validation_y,
            )
            folds.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train_positions),
                    "fold_validation_rows": len(validation_positions),
                    "directory": fold_dir.name,
                    "profiles": profile_stats,
                }
            )
            _LOGGER.info(
                "Exported Dataset B CV fold=%d train_rows=%d fold_validation_rows=%d",
                fold_number,
                len(train_positions),
                len(validation_positions),
            )

        if len(folds) != 5 or not np.all(validation_coverage == 1):
            raise ValueError(
                "Dataset B five fold-validation sets must cover every retained Train row exactly once."
            )
        if expected_semantic_columns != list(TWO_CHANNEL_COLUMNS):
            raise ValueError("Dataset B semantic schema must contain the exact 19 columns.")

        quality = train.quality_audit.as_dict()
        blank_row_audit = {
            "status": quality["status"],
            "input_rows": quality["input_rows"],
            "retained_rows": quality["retained_rows"],
            "rejected_rows": quality["rejected_rows"],
            "rejected_record_ids": quality["rejected_record_ids"],
            "blank_counts": quality["blank_counts"],
        }
        normalization_audit = {
            "raw_1111_counts": quality["raw_1111_counts"],
            "normalized_exempt_counts": quality["normalized_exempt_counts"],
            "raw_equals_normalized": all(
                quality["raw_1111_counts"][column]
                == quality["normalized_exempt_counts"][column]
                for column in EXEMPTION_FLAG_COLUMNS
            ),
            "semantic_1111_count": quality["semantic_1111_count"],
            "row_count_preserved_after_normalization": quality[
                "row_count_preserved_after_normalization"
            ],
            "row_order_preserved": quality["row_order_preserved"],
            "target_alignment_preserved": quality["target_alignment_preserved"],
            "record_id_alignment_preserved": quality["record_id_alignment_preserved"],
        }
        manifest = {
            "dataset": "dataset_b",
            "artifact": "fold_local_preprocessing",
            "run_id": run_id,
            "preprocessing_version": PREPROCESSING_VERSION,
            "split_version": "split_v1",
            "feature_version": loader.feature_version,
            "source_partition": "train",
            "fit_scope": "fold_train_only",
            "use": "cross_validation_only",
            "raw_feature_count": len(FEATURE_COLUMNS),
            "semantic_feature_count": len(TWO_CHANNEL_COLUMNS),
            "encoded_feature_count": ENCODED_FEATURE_COUNT,
            "raw_feature_columns": list(FEATURE_COLUMNS),
            "semantic_columns": expected_semantic_columns,
            "status_reference_categories": dict(STATUS_REFERENCE_CATEGORIES),
            "categorical_normalization": {
                "raw_token": "1111",
                "semantic_token": "Exempt",
                "columns": list(EXEMPTION_FLAG_COLUMNS),
            },
            "blank_row_audit": blank_row_audit,
            "normalization_audit": normalization_audit,
            "tree_profile_shared_by": ["rf", "xgb"],
            "numeric_transforms": {
                "lr": {
                    "log1p_columns": list(LOG1P_VALUE_COLUMNS),
                    "robust_scaled_columns": list(NUMERIC_VALUE_COLUMNS),
                },
                "rf": {"log1p_columns": [], "robust_scaled_columns": []},
                "xgb": {"log1p_columns": [], "robust_scaled_columns": []},
            },
            "matrix_contract": {
                "format": "csr",
                "dtype": "float64",
                "finite": True,
            },
            "sample_weights_exported": False,
            "library_versions": {
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "scipy": version("scipy"),
                "xgboost": version("xgboost"),
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
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"Dataset B preprocessing run already exists: {destination}")
        working.rename(destination)
        final_manifest = destination / "manifest.json"
        _LOGGER.info(
            "Completed Dataset B CV preprocessing run_id=%s folds=%d path=%s",
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
    print(export_cv_preprocessed(args.run_id, args.output_root))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _main()
