"""Object-oriented training orchestration for the locked Set A models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.pipeline.data_loader import SetADataLoader, SetADataSplits
from src.pipeline.evaluate import SetAEvaluator, SetAMetrics
from src.pipeline.preprocess import SetAPreprocessor
from src.utils.config_loader import (
    SetASplitConfig,
    SetATrainingConfig,
    load_set_a_split_config,
    load_set_a_training_config,
)
from src.utils.logger import get_project_logger
from src.utils.path import MODEL_DIR, REPORTS_DIR


LOGGER = get_project_logger(__name__, "set_a_pipeline.log")


@dataclass(frozen=True)
class FittedSetAModel:
    """One fitted model, its frozen validation threshold, and timing evidence."""

    name: str
    estimator: Pipeline
    best_parameters: Mapping[str, Any]
    cv_roc_auc: float
    threshold: float
    training_seconds: float


@dataclass(frozen=True)
class SetAPipelineResult:
    """Artifacts produced by a reproducible Set A pipeline execution."""

    models: Mapping[str, FittedSetAModel]
    metrics: Mapping[str, SetAMetrics]
    comparison: pd.DataFrame
    bootstrap_intervals: pd.DataFrame | None


class SetAModelTrainer:
    """Tune only inside training data, then freeze thresholds on validation."""

    def __init__(self, config: SetATrainingConfig, random_state: int):
        self.config = config
        self.random_state = random_state
        self.evaluator = SetAEvaluator()

    def fit_all(self, splits: SetADataSplits) -> dict[str, FittedSetAModel]:
        splits.validate()
        fitted: dict[str, FittedSetAModel] = {}
        for name in self.config.models:
            fitted[name] = self._fit_one(name, splits)
        return fitted

    def _fit_one(self, name: str, splits: SetADataSplits) -> FittedSetAModel:
        pipeline = Pipeline(
            steps=[("preprocessor", SetAPreprocessor()), ("model", self._build_model(name))]
        )
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=dict(self.config.hyperparameter_grids[name]),
            scoring=self.config.cv_scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            refit=True,
            error_score="raise",
        )
        started = time.perf_counter()
        try:
            search.fit(splits.X_train, splits.y_train)
        except Exception:
            LOGGER.exception("Set A model training failed: model=%s.", name)
            raise
        training_seconds = time.perf_counter() - started
        validation_probability = search.best_estimator_.predict_proba(splits.X_validation)[:, 1]
        threshold = self.evaluator.select_threshold(splits.y_validation, validation_probability)
        fitted = FittedSetAModel(
            name=name,
            estimator=search.best_estimator_,
            best_parameters=dict(search.best_params_),
            cv_roc_auc=float(search.best_score_),
            threshold=threshold,
            training_seconds=training_seconds,
        )
        LOGGER.info(
            "Fitted Set A model: model=%s, cv_roc_auc=%.6f, threshold=%.6f, training_seconds=%.3f, parameters=%s.",
            name,
            fitted.cv_roc_auc,
            fitted.threshold,
            fitted.training_seconds,
            dict(fitted.best_parameters),
        )
        return fitted

    def _build_model(self, name: str) -> ClassifierMixin:
        if name == "logistic_regression":
            return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=self.random_state)
        if name == "random_forest":
            # GridSearchCV parallelizes parameter/CV jobs; keeping the inner
            # forest single-threaded prevents nested process oversubscription.
            return RandomForestClassifier(random_state=self.random_state, n_jobs=1)
        raise ValueError(f"Unsupported locked Set A model: {name}")


class SetAPipelineRunner:
    """Coordinate loading, shared splitting, fitting, comparison, and artifacts."""

    def __init__(
        self,
        split_config: SetASplitConfig | None = None,
        training_config: SetATrainingConfig | None = None,
        data_loader: SetADataLoader | None = None,
    ):
        self.split_config = split_config or load_set_a_split_config()
        self.training_config = training_config or load_set_a_training_config()
        self.data_loader = data_loader or SetADataLoader()
        self.evaluator = SetAEvaluator()

    def run(self, bootstrap: bool = True) -> SetAPipelineResult:
        prepared = self.data_loader.load_prepared_data()
        splits = self.data_loader.split(prepared, self.split_config)
        trainer = SetAModelTrainer(self.training_config, self.split_config.random_state)
        models = trainer.fit_all(splits)

        metrics: dict[str, SetAMetrics] = {}
        probabilities: dict[str, pd.Series] = {}
        calibration_curves: dict[str, pd.DataFrame] = {}
        for name, fitted in models.items():
            inference_started = time.perf_counter()
            probability = fitted.estimator.predict_proba(splits.X_test)[:, 1]
            inference_seconds = time.perf_counter() - inference_started
            probabilities[name] = pd.Series(probability, index=splits.y_test.index, name="p_approved")
            metrics[name] = self.evaluator.evaluate(
                splits.y_test,
                probability,
                fitted.threshold,
                fitted.training_seconds,
                inference_seconds,
            )
            calibration_curves[name] = self.evaluator.calibration_curve(splits.y_test, probability)

        comparison = pd.DataFrame(
            {name: metric.to_dict() for name, metric in metrics.items()}
        ).T.rename_axis("model")
        bootstrap_intervals = None
        if bootstrap:
            bootstrap_intervals = self.evaluator.paired_bootstrap(
                splits.y_test,
                probabilities,
                {name: fitted.threshold for name, fitted in models.items()},
                random_state=self.split_config.random_state,
            )
        result = SetAPipelineResult(models, metrics, comparison, bootstrap_intervals)
        self._write_artifacts(result, calibration_curves)
        self._log_and_print_comparison(comparison)
        return result

    def _write_artifacts(
        self, result: SetAPipelineResult, calibration_curves: Mapping[str, pd.DataFrame]
    ) -> None:
        model_directory = MODEL_DIR / "set_a"
        report_directory = REPORTS_DIR / "set_a"
        model_directory.mkdir(parents=True, exist_ok=True)
        report_directory.mkdir(parents=True, exist_ok=True)
        for name, fitted in result.models.items():
            joblib.dump(_model_artifact_payload(fitted), model_directory / f"{name}.joblib")
        result.comparison.to_csv(report_directory / "model_comparison.csv")
        with (report_directory / "model_metadata.json").open("w", encoding="utf-8") as output:
            json.dump(
                {
                    name: {
                        "best_parameters": dict(fitted.best_parameters),
                        "cv_roc_auc": fitted.cv_roc_auc,
                        "threshold": fitted.threshold,
                        "training_seconds": fitted.training_seconds,
                    }
                    for name, fitted in result.models.items()
                },
                output,
                indent=2,
            )
        for name, curve in calibration_curves.items():
            curve.to_csv(report_directory / f"calibration_{name}.csv", index=False)
        if result.bootstrap_intervals is not None:
            result.bootstrap_intervals.to_csv(report_directory / "paired_bootstrap_intervals.csv", index=False)
        LOGGER.info("Wrote Set A model and metric artifacts to %s and %s.", model_directory, report_directory)

    @staticmethod
    def _log_and_print_comparison(comparison: pd.DataFrame) -> None:
        printable = comparison.round(6).to_string()
        LOGGER.info("Set A model comparison:\n%s", printable)
        print("\nSet A model comparison\n")
        print(printable)


def main() -> None:
    """Run as ``python -m src.pipeline.train`` from the project root."""

    parser = argparse.ArgumentParser(description="Train and compare the locked Set A models.")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip the 1,000-resample paired bootstrap for a quicker development run.",
    )
    arguments = parser.parse_args()
    SetAPipelineRunner().run(bootstrap=not arguments.skip_bootstrap)


def _model_artifact_payload(fitted: FittedSetAModel) -> dict[str, Any]:
    """Create a module-independent joblib payload for a frozen Set A model."""

    return {
        "artifact_type": "set_a_fitted_model",
        "schema_version": 1,
        "name": fitted.name,
        "estimator": fitted.estimator,
        "best_parameters": dict(fitted.best_parameters),
        "cv_roc_auc": fitted.cv_roc_auc,
        "threshold": fitted.threshold,
        "training_seconds": fitted.training_seconds,
    }


if __name__ == "__main__":
    main()
