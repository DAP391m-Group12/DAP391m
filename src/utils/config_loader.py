"""Typed configuration loading for the reproducible Set A pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.path import APP_CONFIG_PATH, MODEL_CONFIG_PATH


@dataclass(frozen=True)
class SetASplitConfig:
    """The single, reproducible split shared by every Set A model."""

    random_state: int
    train_size: float
    validation_size: float
    test_size: float
    stratify_target: bool


@dataclass(frozen=True)
class SetATrainingConfig:
    """Model-selection settings that apply only to the training partition."""

    models: tuple[str, ...]
    cv_folds: int
    cv_scoring: str
    n_jobs: int
    hyperparameter_grids: Mapping[str, Mapping[str, tuple[Any, ...]]]


def load_yaml_config(path: Path) -> Mapping[str, Any]:
    """Read a non-empty YAML mapping with a useful configuration error."""

    try:
        with Path(path).open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file: {path}") from exc
    if not isinstance(config, Mapping):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")
    return config


def load_set_a_split_config(config_path: Path = APP_CONFIG_PATH) -> SetASplitConfig:
    """Load and validate the locked train/validation/test split contract."""

    config = load_yaml_config(config_path)
    try:
        splitting = config["dataset_splitting"]
        proportions = splitting["split_proportions"]
        result = SetASplitConfig(
            random_state=int(splitting["random_state"]),
            train_size=float(proportions["train"]),
            validation_size=float(proportions["validation"]),
            test_size=float(proportions["test"]),
            stratify_target=bool(splitting["stratify_target"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("config.yaml does not contain a complete Set A split contract.") from exc

    if any(size <= 0 or size >= 1 for size in (result.train_size, result.validation_size, result.test_size)):
        raise ValueError("Every Set A split proportion must be between 0 and 1.")
    if abs(result.train_size + result.validation_size + result.test_size - 1.0) > 1e-9:
        raise ValueError("Set A train, validation, and test proportions must sum to 1.")
    if not result.stratify_target:
        raise ValueError("Set A must use a stratified shared split.")
    return result


def load_set_a_training_config(config_path: Path = MODEL_CONFIG_PATH) -> SetATrainingConfig:
    """Load models and search grids for Set A's train-only CV."""

    config = load_yaml_config(config_path)
    try:
        set_a = config["set_a"]
        training = set_a["training"]
        raw_grids = set_a["hyperparameter_grids"]
        models = tuple(set_a["models"])
        result = SetATrainingConfig(
            models=models,
            cv_folds=int(training["cv_folds"]),
            cv_scoring=str(training["cv_scoring"]),
            n_jobs=int(training["n_jobs"]),
            hyperparameter_grids={
                str(model_name): {
                    str(parameter): tuple(values)
                    for parameter, values in grid.items()
                }
                for model_name, grid in raw_grids.items()
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model_config.yaml does not contain a complete Set A training contract.") from exc

    if set_a.get("status") != "locked" or set(result.models) != {"logistic_regression", "random_forest"}:
        raise ValueError("Set A must contain exactly Logistic Regression and Random Forest.")
    if result.cv_folds < 2 or result.cv_scoring != "roc_auc":
        raise ValueError("Set A must use at least two train-only ROC-AUC CV folds.")
    if set(result.hyperparameter_grids) != set(result.models):
        raise ValueError("Set A must define a hyperparameter grid for every configured model.")
    if any(not grid for grid in result.hyperparameter_grids.values()):
        raise ValueError("Set A hyperparameter grids cannot be empty.")
    return result
