"""Train-only full-fit pipeline and artifact validator for PlainMLP v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import yaml
from scipy.sparse import isspmatrix_csr

from src.dataset_b.models.plain_mlp import PlainMLPClassifier
from src.dataset_b.preprocessing import (
    DatasetBSplitLoader,
    make_plain_mlp_preprocessing_pipeline,
)
from src.utils.path import CONFIG_DIR, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1/full_train"
DEFAULT_SELECTION_PATH = (
    PROJECT_ROOT / "artifacts/dataset_b/plain_mlp_v1/configuration_selection.json"
)
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dataset_b/plain_mlp_v1.yaml"
DEFAULT_WRAPPER_PATH = PROJECT_ROOT / "src/dataset_b/models/plain_mlp.py"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = DatasetBSplitLoader.DEFAULT_SPLIT_DIR
DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
EXPECTED_TRAIN_ROWS = 205_310
EXPECTED_FEATURES = 66
CONTRACT_ID = "PMLP-03"
CHECKPOINT_SCOPE = "Train-only full-fit; no Validation or Test access."
PAYLOAD_FILES = {"checkpoints/fitted_bundle.joblib", "full_train_result.json"}
FINAL_FILES = PAYLOAD_FILES | {"manifest.json", "COMPLETED"}


class PlainMLPFullTrainError(RuntimeError):
    """Raised when the PlainMLP full-Train artifact violates its contract."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlainMLPFullTrainError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PlainMLPFullTrainError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PlainMLPFullTrainError("Cannot read the PlainMLP configuration.") from exc
    if not isinstance(value, dict) or value.get("model_version") != "plain_mlp_v1":
        raise PlainMLPFullTrainError("Unexpected PlainMLP configuration.")
    if (
        value.get("status") != "provisional_starting_configuration"
        or value.get("is_final_configuration") is not False
        or value.get("selected_by_train_cv") is not False
    ):
        raise PlainMLPFullTrainError("The immutable source configuration changed status.")
    preprocessing = value.get("preprocessing", {})
    environment = value.get("environment", {})
    class_handling = value.get("class_handling")
    if (
        preprocessing.get("profile") != "plain_mlp_external_ohe"
        or preprocessing.get("encoded_feature_count") != EXPECTED_FEATURES
        or preprocessing.get("stored_format") != "csr"
        or preprocessing.get("stored_dtype") != "float32"
        or environment.get("device") != "cpu"
        or environment.get("torch_build") != "cpu"
        or class_handling
        != {"class_weight": False, "sample_weight": False, "resampling": False}
    ):
        raise PlainMLPFullTrainError("PlainMLP preprocessing, device, or class policy changed.")
    return value


def _validate_runtime(config: dict[str, Any]) -> None:
    environment = config["environment"]
    if (
        platform.python_version() != str(environment.get("python_version"))
        or version("torch").split("+", maxsplit=1)[0]
        != str(environment.get("torch_release"))
        or torch.cuda.is_available()
    ):
        raise PlainMLPFullTrainError("Runtime differs from the locked CPU environment.")


def _validate_selection(path: Path, config_path: Path) -> dict[str, Any]:
    selection = _read_json(path)
    if (
        selection.get("stage") != "configuration_selection_after_train_cv"
        or selection.get("status") != "complete"
        or selection.get("decision") != "accept_for_full_train_refit"
        or selection.get("selected_for_run") is not True
        or selection.get("is_final_configuration_for_run") is not True
        or selection.get("external_validation_used") is not False
        or selection.get("test_used") is not False
        or selection.get("source_configuration_status")
        != "provisional_starting_configuration"
    ):
        raise PlainMLPFullTrainError("PMLP-02 does not authorize a full-Train refit.")
    config = _load_config(config_path)
    hashes = selection.get("source_sha256")
    required_hashes = {
        "plain_mlp_config",
        "neural_preprocessing_manifest",
        "plain_mlp_wrapper",
        "cv_results",
        "cv_oof_predictions",
        "cv_training_history",
    }
    cv_artifact = Path(config.get("contracts", {}).get("cv_artifact", ""))
    if not cv_artifact.is_absolute():
        cv_artifact = PROJECT_ROOT / cv_artifact
    sources = {
        "plain_mlp_config": config_path,
        "neural_preprocessing_manifest": cv_artifact / "manifest.json",
        "plain_mlp_wrapper": DEFAULT_WRAPPER_PATH,
        "cv_results": path.parent / "cv_results.json",
        "cv_oof_predictions": path.parent / "cv_oof_predictions.csv",
        "cv_training_history": path.parent / "cv_training_history.json",
    }
    if not isinstance(hashes, dict) or set(hashes) != required_hashes:
        raise PlainMLPFullTrainError("PMLP-02 source hash coverage is invalid.")
    try:
        mismatches = [name for name, source in sources.items() if _sha256_file(source) != hashes[name]]
    except OSError as exc:
        raise PlainMLPFullTrainError("Cannot read a PMLP-02 source artifact.") from exc
    if mismatches:
        raise PlainMLPFullTrainError(f"PMLP-02 source hash mismatch: {mismatches}")
    if selection.get("locked_configuration") != {
        "architecture": config["architecture"],
        "training": config["training"],
        "class_handling": config["class_handling"],
    }:
        raise PlainMLPFullTrainError("PMLP-02 locked configuration differs from source config.")
    return selection


def _make_estimator(config: dict[str, Any]) -> PlainMLPClassifier:
    architecture = config["architecture"]
    training = config["training"]
    return PlainMLPClassifier(
        hidden_layers=tuple(architecture["hidden_layers"]),
        activation=architecture["activation"],
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        dropout=float(training["dropout"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        random_state=int(training["random_state"]),
        device=config["environment"]["device"],
        deterministic_algorithms=bool(training["deterministic_algorithms"]),
    )


def load_plain_mlp_bundle(path: str | Path) -> tuple[Any, PlainMLPClassifier, dict[str, Any]]:
    """Load a trusted fitted preprocessor and rehydrate its PlainMLP checkpoint."""
    try:
        bundle = joblib.load(Path(path))
    except Exception as exc:
        raise PlainMLPFullTrainError("Cannot load the PlainMLP fitted bundle.") from exc
    if not isinstance(bundle, dict) or set(bundle) != {"preprocessing", "model_checkpoint", "scope"}:
        raise PlainMLPFullTrainError("Invalid PlainMLP fitted bundle structure.")
    scope = bundle["scope"]
    if not isinstance(scope, dict) or scope.get("description") != CHECKPOINT_SCOPE:
        raise PlainMLPFullTrainError("Invalid PlainMLP fitted bundle scope.")
    try:
        estimator = PlainMLPClassifier.from_checkpoint_payload(bundle["model_checkpoint"])
    except Exception as exc:
        raise PlainMLPFullTrainError("Invalid PlainMLP model checkpoint.") from exc
    return bundle["preprocessing"], estimator, scope


def run_plain_mlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    selection_path: str | Path = DEFAULT_SELECTION_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Fit fresh preprocessing and PlainMLP state on full Train and publish atomically."""
    destination = Path(output_dir)
    paths = {
        "selection_path": Path(selection_path),
        "config_path": Path(config_path),
        "split_manifest_path": Path(split_manifest_path),
        "split_dir": Path(split_dir),
        "feature_contract_path": Path(feature_contract_path),
    }
    if destination.exists():
        result = validate_plain_mlp_full_train(destination, **paths)
        return {**result, "resume_action": "SKIPPED_VALID_COMPLETED"}
    working = destination.with_name(destination.name + ".tmp")
    if working.exists():
        raise PlainMLPFullTrainError(
            f"Temporary directory already exists and was preserved: {working}"
        )

    config = _load_config(paths["config_path"])
    _validate_runtime(config)
    selection = _validate_selection(paths["selection_path"], paths["config_path"])
    try:
        loader = DatasetBSplitLoader(
            manifest_path=paths["split_manifest_path"],
            split_dir=paths["split_dir"],
            feature_contract_path=paths["feature_contract_path"],
        )
        partition = loader.load("train")
    except Exception as exc:
        raise PlainMLPFullTrainError("Cannot load the immutable Train partition.") from exc
    record_ids = tuple(partition.record_ids.astype(str).tolist())
    labels = partition.y.to_numpy(dtype=np.int8)
    if (
        len(record_ids) != EXPECTED_TRAIN_ROWS
        or len(set(record_ids)) != EXPECTED_TRAIN_ROWS
        or labels.shape != (EXPECTED_TRAIN_ROWS,)
        or set(np.unique(labels).tolist()) != {0, 1}
    ):
        raise PlainMLPFullTrainError("Train IDs or labels violate the locked split contract.")

    preprocessor = make_plain_mlp_preprocessing_pipeline()
    estimator = _make_estimator(config)
    started = time.perf_counter()
    try:
        transformed = preprocessor.fit_transform(partition.X.reset_index(drop=True))
        if (
            not isspmatrix_csr(transformed)
            or transformed.dtype != np.float32
            or transformed.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURES)
            or not np.isfinite(transformed.data).all()
        ):
            raise PlainMLPFullTrainError("Full-Train preprocessing output is invalid.")
        estimator.fit(transformed, labels)
    except PlainMLPFullTrainError:
        raise
    except Exception as exc:
        raise PlainMLPFullTrainError("PlainMLP full-Train fitting failed.") from exc
    elapsed = time.perf_counter() - started

    sample_rows = min(256, EXPECTED_TRAIN_ROWS)
    before = estimator.predict_proba(transformed[:sample_rows])
    destination.parent.mkdir(parents=True, exist_ok=True)
    working.mkdir()
    (working / "checkpoints").mkdir()
    bundle_path = working / "checkpoints/fitted_bundle.joblib"
    scope = {
        "description": CHECKPOINT_SCOPE,
        "source_partition": "train",
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_accessed": False,
        "test_accessed": False,
        "feature_version": selection["feature_version"],
        "split_version": selection["split_version"],
        "preprocessing_version": selection["preprocessing_version"],
        "selection_sha256": _sha256_file(paths["selection_path"]),
    }
    try:
        joblib.dump(
            {
                "preprocessing": preprocessor,
                "model_checkpoint": estimator.checkpoint_payload(),
                "scope": scope,
            },
            bundle_path,
            compress=3,
        )
        restored_preprocessor, restored_estimator, _ = load_plain_mlp_bundle(bundle_path)
        restored_matrix = restored_preprocessor.transform(
            partition.X.iloc[:sample_rows].reset_index(drop=True)
        )
        after = restored_estimator.predict_proba(restored_matrix)
        if not np.array_equal(before, after):
            raise PlainMLPFullTrainError("Reloaded bundle predictions are not identical.")

        lineage = {
            "configuration_selection_sha256": _sha256_file(paths["selection_path"]),
            "plain_mlp_config_sha256": _sha256_file(paths["config_path"]),
            "split_manifest_sha256": _sha256_file(paths["split_manifest_path"]),
            "train_source_sha256": _sha256_file(paths["split_dir"] / "train.csv"),
            "feature_contract_sha256": _sha256_file(paths["feature_contract_path"]),
            "train_record_ids_sha256": _sha256_ids(record_ids),
        }
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "stage": "fresh_full_train_fit",
            "model_version": "plain_mlp_v1",
            "configuration_source_status": config["status"],
            "configuration_decision": selection["decision"],
            "data_boundary": {
                "source_partition": "train",
                "train_rows": EXPECTED_TRAIN_ROWS,
                "project_validation_accessed": False,
                "project_test_accessed": False,
                "preprocessing_fit_calls": 1,
                "model_fit_calls": 1,
            },
            "matrix": {
                "format": "csr",
                "dtype": "float32",
                "rows": EXPECTED_TRAIN_ROWS,
                "columns": EXPECTED_FEATURES,
                "finite": True,
            },
            "configuration": {
                "architecture": config["architecture"],
                "training": config["training"],
                "class_handling": config["class_handling"],
            },
            "execution": {
                "elapsed_seconds": elapsed,
                "training_history_epochs": len(estimator.training_history_),
                "reload_prediction_rows": sample_rows,
                "reload_predictions_identical": True,
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "scipy": version("scipy"),
                "scikit_learn": version("scikit-learn"),
                "torch": version("torch"),
                "device": estimator.device,
            },
            "lineage_sha256": lineage,
        }
        _write_json(working / "full_train_result.json", result)
        manifest = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "PASS",
            "model_version": "plain_mlp_v1",
            "scope": CHECKPOINT_SCOPE,
            "lineage_sha256": lineage,
            "artifact_sha256": {
                name: _sha256_file(working / name) for name in sorted(PAYLOAD_FILES)
            },
        }
        _write_json(working / "manifest.json", manifest)
        (working / "COMPLETED").write_text(f"{CONTRACT_ID} PASS\n", encoding="utf-8")
        validate_plain_mlp_full_train(working, **paths)
        os.replace(working, destination)
        return validate_plain_mlp_full_train(destination, **paths)
    except PlainMLPFullTrainError:
        raise
    except Exception as exc:
        raise PlainMLPFullTrainError(f"Cannot publish PlainMLP full-Train artifacts: {exc}") from exc


def validate_plain_mlp_full_train(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    selection_path: str | Path = DEFAULT_SELECTION_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate the published bundle, hashes, lineage, scope, and reloadability."""
    directory = Path(output_dir)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    } if directory.is_dir() else set()
    if actual != FINAL_FILES:
        raise PlainMLPFullTrainError("Full-Train artifact set is invalid.")
    if (directory / "COMPLETED").read_text(encoding="utf-8") != f"{CONTRACT_ID} PASS\n":
        raise PlainMLPFullTrainError("Full-Train completion marker is invalid.")
    manifest = _read_json(directory / "manifest.json")
    result = _read_json(directory / "full_train_result.json")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != PAYLOAD_FILES:
        raise PlainMLPFullTrainError("Full-Train artifact hash coverage is invalid.")
    for name, expected in hashes.items():
        if _sha256_file(directory / name) != expected:
            raise PlainMLPFullTrainError(f"Full-Train artifact hash mismatch: {name}")

    config_path = Path(config_path)
    selection_path = Path(selection_path)
    split_manifest_path = Path(split_manifest_path)
    split_dir = Path(split_dir)
    feature_contract_path = Path(feature_contract_path)
    config = _load_config(config_path)
    _validate_runtime(config)
    selection = _validate_selection(selection_path, config_path)
    try:
        loader = DatasetBSplitLoader(
            manifest_path=split_manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        )
        partition = loader.load("train")
        train_ids = tuple(partition.record_ids.astype(str).tolist())
    except Exception as exc:
        raise PlainMLPFullTrainError("Cannot validate immutable Train membership.") from exc
    if len(train_ids) != EXPECTED_TRAIN_ROWS or len(set(train_ids)) != EXPECTED_TRAIN_ROWS:
        raise PlainMLPFullTrainError("Train membership cardinality is invalid.")
    expected_lineage = {
        "configuration_selection_sha256": _sha256_file(selection_path),
        "plain_mlp_config_sha256": _sha256_file(config_path),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "train_source_sha256": _sha256_file(split_dir / "train.csv"),
        "feature_contract_sha256": _sha256_file(feature_contract_path),
        "train_record_ids_sha256": _sha256_ids(train_ids),
    }
    if (
        manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("scope") != CHECKPOINT_SCOPE
        or result.get("contract_id") != CONTRACT_ID
        or result.get("status") != "PASS"
        or result.get("configuration_source_status") != config["status"]
        or result.get("configuration_decision") != selection["decision"]
        or result.get("data_boundary") != {
            "source_partition": "train",
            "train_rows": EXPECTED_TRAIN_ROWS,
            "project_validation_accessed": False,
            "project_test_accessed": False,
            "preprocessing_fit_calls": 1,
            "model_fit_calls": 1,
        }
        or result.get("matrix") != {
            "format": "csr", "dtype": "float32", "rows": EXPECTED_TRAIN_ROWS,
            "columns": EXPECTED_FEATURES, "finite": True,
        }
        or result.get("execution", {}).get("reload_predictions_identical") is not True
        or result.get("lineage_sha256") != expected_lineage
        or manifest.get("lineage_sha256") != expected_lineage
    ):
        raise PlainMLPFullTrainError("Full-Train status, scope, or lineage is invalid.")
    _, estimator, scope = load_plain_mlp_bundle(directory / "checkpoints/fitted_bundle.joblib")
    if (
        estimator.n_features_in_ != EXPECTED_FEATURES
        or len(estimator.training_history_) != int(config["training"]["epochs"])
        or scope.get("train_rows") != EXPECTED_TRAIN_ROWS
        or scope.get("validation_accessed") is not False
        or scope.get("test_accessed") is not False
    ):
        raise PlainMLPFullTrainError("Reloaded bundle metadata is invalid.")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate PlainMLP full-Train fit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = (
            run_plain_mlp_full_train(args.output_dir)
            if args.command == "run"
            else validate_plain_mlp_full_train(args.output_dir)
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except PlainMLPFullTrainError as exc:
        print(f"PlainMLP full-Train failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
