"""Preprocessing for the locked HMDA New York 2024 Set A dataset.

Set A keeps all five applicant-race slots, represents them as an unordered
multi-hot feature set, and maps encoded features back to the seven canonical
explanation groups required for RQ2.  ``action_taken`` is only used to build
the target and is never passed to the model transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_project_logger
from src.utils.path import FEATURES_CONFIG_PATH


LOGGER = get_project_logger(__name__, "preprocess.log")

SET_A_TARGET_NAME = "y_approved"
SET_A_TARGET_SOURCE = "action_taken"
SET_A_RAW_COLUMNS = (
    "applicant_race-1",
    "applicant_race-2",
    "applicant_race-3",
    "applicant_race-4",
    "applicant_race-5",
    "applicant_sex",
    "income",
    "loan_amount",
    "loan_purpose",
    "lien_status",
    "loan_type",
)
SET_A_RACE_COLUMNS = SET_A_RAW_COLUMNS[:5]
SET_A_NUMERICAL_COLUMNS = ("income", "loan_amount")
SET_A_CATEGORICAL_COLUMNS = ("applicant_sex", "loan_purpose", "lien_status", "loan_type")
SET_A_EXPLANATION_SOURCES = {
    "applicant_race": "multihot_group:applicant_race",
    "applicant_sex": "applicant_sex",
    "applicant_income": "income",
    "loan_amount": "loan_amount",
    "loan_purpose": "loan_purpose",
    "lien_status": "lien_status",
    "loan_type": "loan_type",
}


@dataclass(frozen=True)
class SetAFeatureSpec:
    """The data contract read from ``config/features.yaml`` for Set A."""

    target_name: str
    target_source_column: str
    eligible_values: tuple[int, ...]
    positive_values: tuple[int, ...]
    negative_values: tuple[int, ...]
    forbidden_model_inputs: tuple[str, ...]
    raw_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    race_columns: tuple[str, ...]
    explanation_groups: Mapping[str, str]


@dataclass(frozen=True)
class PreparedSetAData:
    """Eligible Set A inputs and the binary approval target."""

    X: pd.DataFrame
    y: pd.Series


class ApplicantRaceMultiHotEncoder(BaseEstimator, TransformerMixin):
    """Encode five HMDA race slots as one unordered multi-hot representation.

    A record receives ``1`` for every declared race category, independently of
    the slot in which that category appeared. Missing slots contribute no
    feature. The encoder learns the union of categories from the fitting data.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "ApplicantRaceMultiHotEncoder":
        frame = self._as_frame(X)
        categories = {
            token
            for value in frame.to_numpy().ravel()
            if (token := self._normalise_value(value)) is not None
        }
        self.categories_ = np.asarray(sorted(categories), dtype=object)
        self.n_features_in_ = frame.shape[1]
        if self.n_features_in_ == 0:
            raise ValueError("Race encoder requires at least one race slot.")
        if not len(self.categories_):
            raise ValueError("Race encoder cannot fit because every race slot is missing.")
        LOGGER.info(
            "Fitted race multi-hot encoder: rows=%d, slots=%d, categories=%d.",
            len(frame),
            self.n_features_in_,
            len(self.categories_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "categories_"):
            raise ValueError("ApplicantRaceMultiHotEncoder must be fitted before transform.")

        frame = self._as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                "Applicant race input has a different number of slots than the fitted data."
            )

        encoded = np.zeros((len(frame), len(self.categories_)), dtype=np.uint8)
        category_index = {category: index for index, category in enumerate(self.categories_)}
        for row_index, row in enumerate(frame.to_numpy()):
            for value in row:
                token = self._normalise_value(value)
                if token in category_index:
                    encoded[row_index, category_index[token]] = 1
        if encoded.shape != (len(frame), len(self.categories_)):
            raise RuntimeError("Race encoder produced an unexpected output shape.")
        LOGGER.info("Transformed race slots: rows=%d, output_features=%d.", *encoded.shape)
        return encoded

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if not hasattr(self, "categories_"):
            raise ValueError("ApplicantRaceMultiHotEncoder must be fitted before naming features.")
        return np.asarray(
            [f"applicant_race__{category}" for category in self.categories_], dtype=object
        )

    @staticmethod
    def _as_frame(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _normalise_value(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, Integral):
            return str(int(value))
        if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
            return str(int(value))

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            return None
        return text


def load_set_a_feature_spec(config_path: Path = FEATURES_CONFIG_PATH) -> SetAFeatureSpec:
    """Load and validate the locked Set A feature contract from YAML."""

    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        target = config["target"]
        dataset = config["datasets"]["dataset_a"]
        race_columns = dataset["multihot_groups"]["applicant_race"]["source_columns"]
        explanation_groups = dataset["explanation_groups"]
    except (KeyError, TypeError) as exc:
        raise ValueError("features.yaml does not contain a valid Set A feature contract.") from exc

    raw_columns = tuple(dataset["model_input_raw_columns"])
    if dataset.get("status") != "locked" or set(explanation_groups) != set(SET_A_EXPLANATION_SOURCES):
        raise ValueError("Set A must be locked and contain exactly seven explanation groups.")
    if not set(race_columns).issubset(raw_columns):
        raise ValueError("Set A race columns must be present in model_input_raw_columns.")

    try:
        spec = SetAFeatureSpec(
            target_name=target["name"],
            target_source_column=target["source_column"],
            eligible_values=tuple(target["eligible_values"]),
            positive_values=tuple(target["positive_values"]),
            negative_values=tuple(target["negative_values"]),
            forbidden_model_inputs=tuple(target["forbidden_model_inputs"]),
            raw_columns=raw_columns,
            numerical_columns=tuple(dataset["numerical_columns"]),
            categorical_columns=tuple(dataset["categorical_columns"]),
            race_columns=tuple(race_columns),
            explanation_groups={
                name: definition["source"] for name, definition in explanation_groups.items()
            },
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("features.yaml does not contain a complete Set A feature contract.") from exc
    _validate_set_a_spec(spec)
    LOGGER.info(
        "Loaded Set A feature specification from %s: raw_columns=%d, explanation_groups=%d.",
        config_path,
        len(spec.raw_columns),
        len(spec.explanation_groups),
    )
    return spec


def prepare_set_a_data(
    frame: pd.DataFrame, spec: SetAFeatureSpec | None = None
) -> PreparedSetAData:
    """Filter eligible decisions and split raw Set A inputs from the target.

    Rows with action codes outside ``{1, 2, 3}`` are outside the research
    population and are excluded. The resulting inputs contain exactly the 11
    locked raw columns; no leakage column is retained.
    """

    spec = spec or load_set_a_feature_spec()
    _validate_set_a_spec(spec)
    _require_columns(frame, (*spec.raw_columns, spec.target_source_column))
    LOGGER.info("Preparing Set A data: input_rows=%d.", len(frame))

    action_taken = pd.to_numeric(frame[spec.target_source_column], errors="coerce")
    eligible = action_taken.isin(spec.eligible_values)
    filtered = frame.loc[eligible]
    if filtered.empty:
        LOGGER.error("Set A preparation found no eligible action_taken values.")
        raise ValueError("No rows have an eligible action_taken value for Set A.")

    y = action_taken.loc[eligible].isin(spec.positive_values).astype("int8")
    y.name = spec.target_name
    X = filtered.loc[:, list(spec.raw_columns)].copy()
    if len(X) != len(y):
        raise RuntimeError("Set A inputs and target must contain the same rows.")
    if tuple(X.columns) != spec.raw_columns:
        raise RuntimeError("Set A inputs must retain the locked column order.")
    if set(spec.forbidden_model_inputs).intersection(X.columns):
        raise RuntimeError("A forbidden Set A column was retained as a model input.")
    if y.name != spec.target_name or not y.isin((0, 1)).all():
        raise RuntimeError("Prepared Set A target must be named and binary.")
    LOGGER.info(
        "Prepared Set A data: eligible_rows=%d, approved=%d, denied=%d.",
        len(X),
        int(y.sum()),
        int((y == 0).sum()),
    )
    return PreparedSetAData(X=X, y=y)


class SetAPreprocessor(BaseEstimator, TransformerMixin):
    """Fit/transform the model inputs and retain encoded-to-group provenance."""

    def __init__(self, spec: SetAFeatureSpec | None = None):
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SetAPreprocessor":
        self.spec_ = self.spec or load_set_a_feature_spec()
        _validate_set_a_spec(self.spec_)
        _require_columns(X, self.spec_.raw_columns)
        LOGGER.info("Fitting Set A preprocessor: input_rows=%d, input_columns=%d.", len(X), X.shape[1])
        self.transformer_ = _build_column_transformer(self.spec_)
        self.transformer_.fit(_normalise_numeric_columns(X, self.spec_), y)
        self.feature_names_out_ = tuple(self.transformer_.get_feature_names_out())
        self.explanation_feature_groups_ = _encoded_feature_groups(
            self.feature_names_out_, self.spec_
        )
        _validate_encoded_output(
            self.feature_names_out_, self.explanation_feature_groups_, self.spec_
        )
        LOGGER.info(
            "Fitted Set A preprocessor: encoded_features=%d, explanation_groups=%d.",
            len(self.feature_names_out_),
            len(self.explanation_feature_groups_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "transformer_"):
            raise ValueError("SetAPreprocessor must be fitted before transform.")
        _require_columns(X, self.spec_.raw_columns)
        transformed = self.transformer_.transform(_normalise_numeric_columns(X, self.spec_))
        if transformed.shape != (len(X), len(self.feature_names_out_)):
            raise RuntimeError("Set A preprocessing produced an unexpected output shape.")
        result = pd.DataFrame(
            transformed,
            columns=self.feature_names_out_,
            index=X.index,
        )
        if not result.index.equals(X.index):
            raise RuntimeError("Preprocessing must preserve the input row index.")
        LOGGER.info(
            "Transformed Set A data: input_rows=%d, output_columns=%d.",
            len(X),
            result.shape[1],
        )
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("SetAPreprocessor must be fitted before naming features.")
        return np.asarray(self.feature_names_out_, dtype=object)

    def explanation_groups(self) -> Mapping[str, tuple[str, ...]]:
        """Return every encoded feature assigned to one canonical RQ2 group."""

        if not hasattr(self, "explanation_feature_groups_"):
            raise ValueError("SetAPreprocessor must be fitted before requesting groups.")
        return self.explanation_feature_groups_.copy()


def _build_column_transformer(spec: SetAFeatureSpec) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("race", ApplicantRaceMultiHotEncoder(), list(spec.race_columns)),
            ("numeric", numeric_pipeline, list(spec.numerical_columns)),
            ("categorical", categorical_pipeline, list(spec.categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def _encoded_feature_groups(
    feature_names: Sequence[str], spec: SetAFeatureSpec
) -> dict[str, tuple[str, ...]]:
    group_features: dict[str, list[str]] = {group: [] for group in spec.explanation_groups}
    categorical_to_group = {
        "applicant_sex": "applicant_sex",
        "loan_purpose": "loan_purpose",
        "lien_status": "lien_status",
        "loan_type": "loan_type",
    }
    numeric_to_group = {"income": "applicant_income", "loan_amount": "loan_amount"}

    for name in feature_names:
        if name.startswith("race__"):
            group_features["applicant_race"].append(name)
            continue
        if name.startswith("numeric__"):
            source_column = name.removeprefix("numeric__")
            group_features[numeric_to_group[source_column]].append(name)
            continue
        if name.startswith("categorical__"):
            encoded_name = name.removeprefix("categorical__")
            source_column = next(
                (column for column in categorical_to_group if encoded_name.startswith(f"{column}_")),
                None,
            )
            if source_column is not None:
                group_features[categorical_to_group[source_column]].append(name)
                continue
        raise ValueError(f"Cannot map encoded feature {name!r} to a Set A explanation group.")

    return {group: tuple(features) for group, features in group_features.items()}


def _validate_set_a_spec(spec: SetAFeatureSpec) -> None:
    """Validate non-negotiable constraints of the locked Set A contract.

    These are explicit exceptions rather than ``assert`` statements so the
    research contract remains enforced when Python is run with ``-O``.
    """

    checks = (
        (spec.target_name == SET_A_TARGET_NAME, "Set A target must be y_approved."),
        (spec.target_source_column == SET_A_TARGET_SOURCE, "Set A target source must be action_taken."),
        (spec.raw_columns == SET_A_RAW_COLUMNS, "Set A must retain the locked 11 raw model inputs and order."),
        (spec.race_columns == SET_A_RACE_COLUMNS, "Set A must retain all five locked applicant-race slots."),
        (spec.numerical_columns == SET_A_NUMERICAL_COLUMNS, "Set A numerical columns must match the locked contract."),
        (spec.categorical_columns == SET_A_CATEGORICAL_COLUMNS, "Set A categorical columns must match the locked contract."),
        (set(spec.eligible_values) == {1, 2, 3}, "Set A eligible action codes must be exactly {1, 2, 3}."),
        (set(spec.positive_values) == {1, 2}, "Set A approval action codes must be exactly {1, 2}."),
        (set(spec.negative_values) == {3}, "Set A denial action codes must be exactly {3}."),
        (not set(spec.positive_values).intersection(spec.negative_values), "Set A positive and negative action codes must not overlap."),
        (set(spec.positive_values).union(spec.negative_values) == set(spec.eligible_values), "Set A eligible action codes must partition into approval and denial."),
        (set(spec.forbidden_model_inputs) == {SET_A_TARGET_SOURCE}, "Set A must forbid action_taken as a model input."),
        (spec.target_source_column not in spec.raw_columns, "action_taken must not be a model input."),
        (dict(spec.explanation_groups) == SET_A_EXPLANATION_SOURCES, "Set A explanation groups must match the locked contract."),
    )
    for condition, message in checks:
        if not condition:
            raise ValueError(message)

    routes = (set(spec.race_columns), set(spec.numerical_columns), set(spec.categorical_columns))
    if any(left.intersection(right) for index, left in enumerate(routes) for right in routes[index + 1 :]):
        raise ValueError("Every Set A raw input must have exactly one preprocessing route.")
    if set().union(*routes) != set(spec.raw_columns):
        raise ValueError("Every Set A raw input must have exactly one preprocessing route.")


def _validate_encoded_output(
    feature_names: Sequence[str],
    feature_groups: Mapping[str, tuple[str, ...]],
    spec: SetAFeatureSpec,
) -> None:
    """Validate the RQ2 grouping remains a one-to-one mapping after encoding."""

    grouped_features = [feature for features in feature_groups.values() for feature in features]
    if set(feature_groups) != set(spec.explanation_groups):
        raise RuntimeError("Encoded features must retain all canonical explanation groups.")
    if any(not features for features in feature_groups.values()):
        raise ValueError("Every canonical Set A explanation group must retain at least one encoded feature.")
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("Encoded feature names must be unique.")
    if set(grouped_features) != set(feature_names):
        raise RuntimeError("Every encoded feature must belong to exactly one explanation group.")
    if len(grouped_features) != len(set(grouped_features)):
        raise RuntimeError("An encoded feature cannot belong to multiple explanation groups.")


def _normalise_numeric_columns(frame: pd.DataFrame, spec: SetAFeatureSpec) -> pd.DataFrame:
    """Coerce numeric HMDA fields and treat non-finite values as missing."""

    normalised = frame.loc[:, list(spec.raw_columns)].copy()
    for column in spec.numerical_columns:
        values = pd.to_numeric(normalised[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        normalised[column] = values
    return normalised


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Set A preprocessing expects a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Set A input columns must be unique.")
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        LOGGER.error("Set A input is missing required columns: %s.", ", ".join(missing))
        raise ValueError(f"Missing required Set A columns: {', '.join(missing)}")
