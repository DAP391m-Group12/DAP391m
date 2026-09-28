"""Export fold-local Dataset B preprocessing for notebook CV experiments."""

from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz

from src.dataset_b.splits import DatasetBSplitter
from src.utils.path import PROJECT_ROOT

from .categorical_encoder import ENCODED_FEATURE_COUNT, STATUS_REFERENCE_CATEGORIES
from .profiles import make_preprocessing_pipeline
from .split_inputs import DatasetBSplitLoader


_LOGGER = logging.getLogger(__name__)
PREPROCESSING_VERSION = "dataset_b_preprocessing_v2"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "preprocessing" / "dataset_b_v1" / "cv_preprocessing_v2"
)


def export_cv_preprocessed(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write five fold-specific LR and shared RF/XGBoost matrices.

    Only the existing Train partition is read. Each profile is fitted on its
    fold-train rows; fold-validation is transformed with that fitted profile.
    The exported matrices must not be treated as a full-Train refit artifact.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Dataset B preprocessing export already exists: {destination}")

    loader = DatasetBSplitLoader()
    train = loader.load("train")
    splitter = DatasetBSplitter.from_yaml().cv_splitter()
    destination.mkdir(parents=True)

    folds: list[dict[str, object]] = []
    expected_names: list[str] | None = None
    expected_sources: list[str] | None = None
    expected_categories: list[str | None] | None = None
    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(train.X, train.y), start=1
    ):
        fold_dir = destination / f"fold_{fold_number:02d}"
        fold_dir.mkdir()
        X_train = train.X.iloc[train_positions]
        X_validation = train.X.iloc[validation_positions]
        profile_stats: dict[str, object] = {}

        for export_name, profile_name in (("lr", "lr"), ("tree", "rf")):
            profile = make_preprocessing_pipeline(profile_name)
            train_matrix = profile.fit_transform(X_train, train.y.iloc[train_positions])
            validation_matrix = profile.transform(X_validation)
            if (
                train_matrix.shape != (len(train_positions), ENCODED_FEATURE_COUNT)
                or validation_matrix.shape != (len(validation_positions), ENCODED_FEATURE_COUNT)
            ):
                raise ValueError("Dataset B CV preprocessing produced an unexpected matrix shape.")
            save_npz(fold_dir / f"{export_name}_train.npz", train_matrix)
            save_npz(fold_dir / f"{export_name}_fold_validation.npz", validation_matrix)

            names = profile.get_feature_names_out().tolist()
            encoder = profile.named_steps["encode"]
            sources = list(encoder.feature_sources_)
            categories = list(encoder.feature_categories_)
            if expected_names is None:
                expected_names, expected_sources, expected_categories = names, sources, categories
            elif (names, sources, categories) != (expected_names, expected_sources, expected_categories):
                raise ValueError("Dataset B CV encoded column contract changed between folds or profiles.")

            stats: dict[str, object] = {
                "numeric_medians": dict(profile.named_steps["two_channel"].imputer_.medians_),
            }
            if profile_name == "lr":
                scaler = profile.named_steps["scale_numeric"].scaler_
                stats["robust_center"] = scaler.center_.tolist()
                stats["robust_scale"] = scaler.scale_.tolist()
            profile_stats[export_name] = stats

        np.savez_compressed(
            fold_dir / "rows.npz",
            train_positions=train_positions,
            fold_validation_positions=validation_positions,
            train_record_ids=train.record_ids.iloc[train_positions].to_numpy(dtype=str),
            fold_validation_record_ids=train.record_ids.iloc[validation_positions].to_numpy(dtype=str),
            train_y=train.y.iloc[train_positions].to_numpy(dtype=np.int8),
            fold_validation_y=train.y.iloc[validation_positions].to_numpy(dtype=np.int8),
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
            fold_number, len(train_positions), len(validation_positions),
        )

    manifest = {
        "dataset": "dataset_b",
        "artifact": "fold_local_preprocessing",
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_version": "split_v1",
        "feature_version": loader.feature_version,
        "source_partition": "train",
        "fit_scope": "fold_train_only",
        "use": "cross_validation_only",
        "encoded_feature_count": ENCODED_FEATURE_COUNT,
        "status_reference_categories": dict(STATUS_REFERENCE_CATEGORIES),
        "tree_profile_shared_by": ["rf", "xgb"],
        "numeric_scaling": {"lr": "RobustScaler", "rf": "none", "xgb": "none"},
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
        "folds": folds,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info("Completed Dataset B CV preprocessing export folds=%d path=%s", len(folds), manifest_path)
    return manifest_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    print(export_cv_preprocessed())
