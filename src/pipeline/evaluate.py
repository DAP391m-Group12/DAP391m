"""Set A evaluation, threshold selection, and paired bootstrap intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_project_logger


LOGGER = get_project_logger(__name__, "set_a_pipeline.log")


@dataclass(frozen=True)
class SetAMetrics:
    """Metrics required to compare the two Set A models."""

    roc_auc_approved: float
    pr_auc_denied: float
    brier_score_approved: float
    calibration_intercept: float
    calibration_slope: float
    balanced_accuracy: float
    precision_approved: float
    recall_approved: float
    f1_approved: float
    precision_denied: float
    recall_denied: float
    f1_denied: float
    threshold: float
    training_seconds: float
    inference_seconds: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class SetAEvaluator:
    """Evaluate probabilities for `P(approved)` without calibrating them."""

    metric_names = (
        "roc_auc_approved",
        "pr_auc_denied",
        "brier_score_approved",
        "balanced_accuracy",
        "precision_approved",
        "recall_approved",
        "f1_approved",
        "precision_denied",
        "recall_denied",
        "f1_denied",
    )

    @staticmethod
    def select_threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
        """Select and freeze the validation threshold maximizing balanced accuracy."""

        y = _as_binary_target(y_true)
        probabilities_array = _as_probabilities(probabilities)
        _, _, thresholds = roc_curve(y, probabilities_array)
        candidates = np.unique(np.concatenate(([0.0, 1.0], thresholds[np.isfinite(thresholds)])))
        scores = np.asarray(
            [balanced_accuracy_score(y, probabilities_array >= threshold) for threshold in candidates]
        )
        best_score = scores.max()
        # A higher threshold is deterministic and reduces false approvals when
        # several thresholds have identical validation balanced accuracy.
        threshold = float(candidates[scores == best_score].max())
        LOGGER.info(
            "Selected frozen Set A validation threshold: threshold=%.6f, balanced_accuracy=%.6f.",
            threshold,
            best_score,
        )
        return threshold

    def evaluate(
        self,
        y_true: Sequence[int],
        probabilities: Sequence[float],
        threshold: float,
        training_seconds: float,
        inference_seconds: float,
    ) -> SetAMetrics:
        y = _as_binary_target(y_true)
        probability = _as_probabilities(probabilities)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Set A threshold must be between 0 and 1.")
        predicted = (probability >= threshold).astype(np.int8)
        intercept, slope = _calibration_coefficients(y, probability)
        metrics = SetAMetrics(
            roc_auc_approved=float(roc_auc_score(y, probability)),
            pr_auc_denied=float(average_precision_score(1 - y, 1 - probability)),
            brier_score_approved=float(brier_score_loss(y, probability)),
            calibration_intercept=intercept,
            calibration_slope=slope,
            balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
            precision_approved=float(precision_score(y, predicted, pos_label=1, zero_division=0)),
            recall_approved=float(recall_score(y, predicted, pos_label=1, zero_division=0)),
            f1_approved=float(f1_score(y, predicted, pos_label=1, zero_division=0)),
            precision_denied=float(precision_score(y, predicted, pos_label=0, zero_division=0)),
            recall_denied=float(recall_score(y, predicted, pos_label=0, zero_division=0)),
            f1_denied=float(f1_score(y, predicted, pos_label=0, zero_division=0)),
            threshold=float(threshold),
            training_seconds=float(training_seconds),
            inference_seconds=float(inference_seconds),
        )
        LOGGER.info(
            "Evaluated Set A model: roc_auc=%.6f, denied_pr_auc=%.6f, balanced_accuracy=%.6f.",
            metrics.roc_auc_approved,
            metrics.pr_auc_denied,
            metrics.balanced_accuracy,
        )
        return metrics

    @staticmethod
    def calibration_curve(y_true: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> pd.DataFrame:
        """Return a reporting artifact; probabilities remain unadjusted."""

        y = _as_binary_target(y_true)
        probability = _as_probabilities(probabilities)
        observed, predicted = calibration_curve(y, probability, n_bins=bins, strategy="quantile")
        return pd.DataFrame({"mean_predicted_probability": predicted, "observed_approval_rate": observed})

    def paired_bootstrap(
        self,
        y_true: Sequence[int],
        model_probabilities: Mapping[str, Sequence[float]],
        thresholds: Mapping[str, float],
        resamples: int = 1000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute percentile CIs with the same test-row resamples for all models."""

        y = _as_binary_target(y_true)
        if resamples < 1:
            raise ValueError("Set A paired bootstrap needs at least one resample.")
        probabilities = {name: _as_probabilities(values) for name, values in model_probabilities.items()}
        if set(probabilities) != set(thresholds):
            raise ValueError("Every Set A bootstrap model must have probabilities and a frozen threshold.")
        if any(len(values) != len(y) for values in probabilities.values()):
            raise ValueError("Set A bootstrap probabilities must align with the test target.")

        samples: dict[str, dict[str, list[float]]] = {
            name: {metric: [] for metric in self.metric_names} for name in probabilities
        }
        random = np.random.default_rng(random_state)
        for _ in range(resamples):
            indices = random.integers(0, len(y), size=len(y))
            sample_y = y[indices]
            # Stratified test data makes one-class bootstrap draws vanishingly
            # unlikely, but retain a safe guard for small test fixtures.
            if np.unique(sample_y).size < 2:
                continue
            for name, probability in probabilities.items():
                sample_metrics = _bootstrap_metric_values(
                    sample_y,
                    probability[indices],
                    thresholds[name],
                )
                for metric, value in sample_metrics.items():
                    samples[name][metric].append(value)

        rows = []
        for name, metric_samples in samples.items():
            for metric, values in metric_samples.items():
                if not values:
                    raise ValueError("Set A bootstrap produced no valid binary resamples.")
                rows.append(
                    {
                        "model": name,
                        "metric": metric,
                        "ci_lower": float(np.quantile(values, 0.025)),
                        "ci_upper": float(np.quantile(values, 0.975)),
                        "resamples": len(values),
                    }
                )
        LOGGER.info("Completed paired Set A bootstrap: models=%d, resamples=%d.", len(probabilities), resamples)
        return pd.DataFrame(rows)


def _as_binary_target(y_true: Sequence[int]) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int8)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, (0, 1)).all() or np.unique(y).size != 2:
        raise ValueError("Set A evaluation requires a non-empty binary target containing both classes.")
    return y


def _as_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Set A model probabilities must be finite values in [0, 1].")
    return probability


def _calibration_coefficients(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return float(calibration_model.intercept_[0]), float(calibration_model.coef_[0, 0])


def _bootstrap_metric_values(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute resampled RQ1 comparison metrics without fitting calibration models."""

    predicted = (probability >= threshold).astype(np.int8)
    return {
        "roc_auc_approved": float(roc_auc_score(y, probability)),
        "pr_auc_denied": float(average_precision_score(1 - y, 1 - probability)),
        "brier_score_approved": float(brier_score_loss(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision_approved": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "recall_approved": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "f1_approved": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "precision_denied": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        "recall_denied": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "f1_denied": float(f1_score(y, predicted, pos_label=0, zero_division=0)),
    }
