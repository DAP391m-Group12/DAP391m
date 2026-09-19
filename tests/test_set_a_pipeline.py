import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.pipeline.data_loader import SetADataLoader
from src.pipeline.evaluate import SetAEvaluator
from src.pipeline.preprocess import prepare_set_a_data
from src.pipeline.predict import SetAPredictor
from src.pipeline.train import SetAModelTrainer, _model_artifact_payload
from src.utils.config_loader import SetASplitConfig, SetATrainingConfig


def _set_a_frame(rows: int = 80) -> pd.DataFrame:
    actions = np.resize([1, 2, 3], rows)
    categories = np.arange(rows)
    return pd.DataFrame(
        {
            "action_taken": actions,
            "applicant_race-1": np.resize([1, 2, 3], rows),
            "applicant_race-2": np.resize([2, np.nan, np.nan], rows),
            "applicant_race-3": np.nan,
            "applicant_race-4": np.nan,
            "applicant_race-5": np.nan,
            "applicant_sex": np.resize([1, 2], rows),
            "income": 50.0 + categories,
            "loan_amount": 100.0 + 2 * categories,
            "loan_purpose": np.resize([1, 2], rows),
            "lien_status": np.resize([1, 2], rows),
            "loan_type": np.resize([1, 2], rows),
        }
    )


class TestSetAPipeline(unittest.TestCase):
    def setUp(self):
        self.prepared = prepare_set_a_data(_set_a_frame())
        self.split_config = SetASplitConfig(
            random_state=42,
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            stratify_target=True,
        )

    def test_shared_split_is_stratified_reproducible_and_disjoint(self):
        first = SetADataLoader.split(self.prepared, self.split_config)
        second = SetADataLoader.split(self.prepared, self.split_config)

        self.assertTrue(first.X_train.index.equals(second.X_train.index))
        self.assertEqual(len(first.X_train), 48)
        self.assertEqual(len(first.X_validation), 16)
        self.assertEqual(len(first.X_test), 16)
        self.assertFalse(set(first.X_train.index).intersection(first.X_validation.index))
        self.assertFalse(set(first.X_train.index).intersection(first.X_test.index))
        self.assertFalse(set(first.X_validation.index).intersection(first.X_test.index))
        for target in (first.y_train, first.y_validation, first.y_test):
            self.assertEqual(set(target.unique()), {0, 1})

    def test_evaluator_selects_threshold_and_returns_comparison_metrics(self):
        evaluator = SetAEvaluator()
        y = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.10, 0.35, 0.70, 0.90])

        threshold = evaluator.select_threshold(y, probability)
        metrics = evaluator.evaluate(y, probability, threshold, 1.0, 0.1)
        intervals = evaluator.paired_bootstrap(
            y,
            {"logistic_regression": probability, "random_forest": probability},
            {"logistic_regression": threshold, "random_forest": threshold},
            resamples=10,
            random_state=42,
        )

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertEqual(metrics.roc_auc_approved, 1.0)
        self.assertEqual(metrics.balanced_accuracy, 1.0)
        self.assertEqual(set(intervals["model"]), {"logistic_regression", "random_forest"})
        self.assertEqual(len(intervals), 20)

    def test_trainer_keeps_preprocessing_inside_train_only_cv_pipeline(self):
        splits = SetADataLoader.split(self.prepared, self.split_config)
        training_config = SetATrainingConfig(
            models=("logistic_regression",),
            cv_folds=2,
            cv_scoring="roc_auc",
            n_jobs=1,
            hyperparameter_grids={
                "logistic_regression": {"model__C": (1.0,), "model__class_weight": (None,)}
            },
        )

        fitted = SetAModelTrainer(training_config, random_state=42).fit_all(splits)["logistic_regression"]

        self.assertIn("preprocessor", fitted.estimator.named_steps)
        self.assertIn("model", fitted.estimator.named_steps)
        self.assertGreaterEqual(fitted.threshold, 0.0)
        self.assertLessEqual(fitted.threshold, 1.0)
        self.assertGreaterEqual(fitted.cv_roc_auc, 0.0)

    def test_predictor_loads_a_module_independent_frozen_artifact(self):
        splits = SetADataLoader.split(self.prepared, self.split_config)
        training_config = SetATrainingConfig(
            models=("logistic_regression",),
            cv_folds=2,
            cv_scoring="roc_auc",
            n_jobs=1,
            hyperparameter_grids={
                "logistic_regression": {"model__C": (1.0,), "model__class_weight": (None,)}
            },
        )
        fitted = SetAModelTrainer(training_config, random_state=42).fit_all(splits)["logistic_regression"]

        with TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "logistic_regression.joblib"
            joblib.dump(_model_artifact_payload(fitted), artifact_path)
            predictions = SetAPredictor("logistic_regression", Path(temporary_directory)).load().predict(
                splits.X_test
            )

        self.assertEqual(predictions.columns.tolist(), ["p_approved", "predicted_approved"])
        self.assertEqual(len(predictions), len(splits.X_test))
        self.assertTrue(predictions["p_approved"].between(0, 1).all())
