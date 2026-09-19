"""Loading and serving frozen Set A probability models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.pipeline.preprocess import load_set_a_feature_spec
from src.pipeline.train import FittedSetAModel
from src.utils.logger import get_project_logger
from src.utils.path import MODEL_DIR


LOGGER = get_project_logger(__name__, "set_a_pipeline.log")


class SetAPredictor:
    """Load one frozen Set A artifact and return `P(approved)` plus its decision."""

    def __init__(self, model_name: str, model_directory: Path = MODEL_DIR / "set_a"):
        self.model_name = model_name
        self.model_path = Path(model_directory) / f"{model_name}.joblib"
        self.fitted_model: FittedSetAModel | None = None

    def load(self) -> "SetAPredictor":
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Set A model artifact was not found: {self.model_path}")
        artifact: Mapping[str, Any] = joblib.load(self.model_path)
        if not isinstance(artifact, Mapping) or artifact.get("artifact_type") != "set_a_fitted_model":
            raise ValueError(f"Invalid Set A model artifact: {self.model_path}")
        if artifact.get("name") != self.model_name:
            raise ValueError(f"Set A model artifact name does not match requested model: {self.model_path}")
        try:
            self.fitted_model = FittedSetAModel(
                name=str(artifact["name"]),
                estimator=artifact["estimator"],
                best_parameters=dict(artifact["best_parameters"]),
                cv_roc_auc=float(artifact["cv_roc_auc"]),
                threshold=float(artifact["threshold"]),
                training_seconds=float(artifact["training_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Set A model artifact: {self.model_path}") from exc
        LOGGER.info("Loaded frozen Set A model artifact: model=%s, path=%s.", self.model_name, self.model_path)
        return self

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_model is None:
            raise ValueError("SetAPredictor must load a model before predicting.")
        spec = load_set_a_feature_spec()
        missing = sorted(set(spec.raw_columns).difference(inputs.columns))
        if missing:
            raise ValueError(f"Set A prediction inputs are missing columns: {', '.join(missing)}")
        probability = self.fitted_model.estimator.predict_proba(inputs.loc[:, list(spec.raw_columns)])[:, 1]
        result = pd.DataFrame(
            {
                "p_approved": probability,
                "predicted_approved": (probability >= self.fitted_model.threshold).astype("int8"),
            },
            index=inputs.index,
        )
        LOGGER.info("Generated Set A predictions: model=%s, rows=%d.", self.model_name, len(result))
        return result
