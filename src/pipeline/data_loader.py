"""Data access and shared splitting for the locked Set A study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline.preprocess import (
    PreparedSetAData,
    load_set_a_feature_spec,
    prepare_set_a_data,
)
from src.utils.config_loader import SetASplitConfig
from src.utils.logger import get_project_logger
from src.utils.path import RAW_STATE_NY_PATH


LOGGER = get_project_logger(__name__, "set_a_pipeline.log")


@dataclass(frozen=True)
class SetADataSplits:
    """The one stratified raw-input split shared by all Set A models."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def validate(self) -> None:
        partitions = (
            ("train", self.X_train, self.y_train),
            ("validation", self.X_validation, self.y_validation),
            ("test", self.X_test, self.y_test),
        )
        indices: set[object] = set()
        for name, X, y in partitions:
            if X.empty or y.empty or len(X) != len(y):
                raise ValueError(f"Set A {name} split must contain aligned input rows and targets.")
            if not X.index.equals(y.index):
                raise ValueError(f"Set A {name} inputs and target must retain the same index.")
            if not y.isin((0, 1)).all() or y.nunique() != 2:
                raise ValueError(f"Set A {name} split must retain both binary target classes.")
            overlap = indices.intersection(X.index)
            if overlap:
                raise ValueError("Set A train, validation, and test rows must not overlap.")
            indices.update(X.index)


class SetADataLoader:
    """Load only the columns needed by Set A and create its shared split."""

    def __init__(self, raw_data_path: Path = RAW_STATE_NY_PATH):
        self.raw_data_path = Path(raw_data_path)

    def load_prepared_data(self) -> PreparedSetAData:
        if not self.raw_data_path.is_file():
            raise FileNotFoundError(f"Set A raw data was not found: {self.raw_data_path}")

        spec = load_set_a_feature_spec()
        frame = pd.read_csv(
            self.raw_data_path,
            usecols=[spec.target_source_column, *spec.raw_columns],
            low_memory=False,
        )
        prepared = prepare_set_a_data(frame, spec)
        LOGGER.info(
            "Loaded Set A raw data: path=%s, raw_rows=%d, eligible_rows=%d.",
            self.raw_data_path,
            len(frame),
            len(prepared.X),
        )
        return prepared

    @staticmethod
    def split(prepared: PreparedSetAData, config: SetASplitConfig) -> SetADataSplits:
        """Create a deterministic 60/20/20-style stratified shared split."""

        X_train, X_remaining, y_train, y_remaining = train_test_split(
            prepared.X,
            prepared.y,
            train_size=config.train_size,
            random_state=config.random_state,
            stratify=prepared.y,
        )
        validation_fraction_of_remaining = config.validation_size / (
            config.validation_size + config.test_size
        )
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_remaining,
            y_remaining,
            train_size=validation_fraction_of_remaining,
            random_state=config.random_state,
            stratify=y_remaining,
        )
        splits = SetADataSplits(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_test=X_test,
            y_test=y_test,
        )
        splits.validate()
        LOGGER.info(
            "Created shared Set A split: train=%d, validation=%d, test=%d, seed=%d.",
            len(splits.X_train),
            len(splits.X_validation),
            len(splits.X_test),
            config.random_state,
        )
        return splits
