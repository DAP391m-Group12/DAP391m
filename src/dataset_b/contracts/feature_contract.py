"""Semantic feature-contract and immutable split checks shared by neural routes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from src.dataset_b.preprocessing import DatasetBSplitLoader, SplitInputError
from src.utils.path import CONFIG_DIR, DATASET_B_DIR, PROJECT_ROOT


DEFAULT_FEATURE_CONTRACT_PATH = CONFIG_DIR / "dataset_b/feature_set_v1.yaml"
DEFAULT_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts/dataset_b/split_v1.json"
DEFAULT_SPLIT_DIR = DATASET_B_DIR / "split_v1"

LOCKED_SPLIT_MANIFEST_SHA256 = (
    "66816ba47da64bfd85c010a90a14b9d80443a46d319258bf9fc1b3657f731596"
)
EXPECTED_PARTITION_ROWS: Mapping[str, int] = {
    "train": 205_310,
    "validation": 43_995,
    "test": 43_996,
}

EXPECTED_MODEL_INPUTS = (
    "loan_type",
    "loan_purpose",
    "lien_status",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "income",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "occupancy_type",
    "preapproval",
    "interest_only_payment",
    "balloon_payment",
)
EXPECTED_AUDIT_COLUMNS = (
    "lei",
    "county_code",
    "applicant_race-1",
    "applicant_race-2",
    "applicant_race-3",
    "applicant_race-4",
    "applicant_race-5",
    "applicant_sex",
    "applicant_age",
)

_EXPECTED_ROOT_KEYS = {
    "schema_version",
    "dataset",
    "feature_version",
    "status",
    "purpose",
    "source_of_truth",
    "target",
    "model_input_raw_columns",
    "audit_only_columns",
    "protected_attributes",
    "constraints",
}
_EXPECTED_METADATA = {
    "schema_version": 1,
    "dataset": "dataset_b",
    "feature_version": "feature_set_v1",
    "status": "final_feature_lock",
    "purpose": "final_model_retraining",
}
_EXPECTED_SOURCE_OF_TRUTH = {
    "document": "docs/handbook/contracts/feature_governance.md",
    "section": 11,
}
_EXPECTED_TARGET = {
    "name": "y_approved",
    "source_column": "action_taken",
    "eligible_values": [1, 2, 3],
    "positive_values": [1, 2],
    "negative_values": [3],
    "forbidden_model_inputs": ["action_taken"],
}
_EXPECTED_PROTECTED_ATTRIBUTES = {
    "sex": {"source_column": "applicant_sex"},
    "applicant_race": {
        "source_columns": [
            "applicant_race-1",
            "applicant_race-2",
            "applicant_race-3",
            "applicant_race-4",
            "applicant_race-5",
        ]
    },
}
_EXPECTED_CONSTRAINTS = {
    "audit_columns_are_model_inputs": False,
    "final_model_features": True,
    "raw_feature_count": 15,
    "semantic_feature_count": 19,
    "preprocessing_profile": "dataset_b_preprocessing_v2",
    "encoded_feature_count": 62,
    "numeric_status_reference_category": "OBSERVED",
    "missing_category_retained": False,
}
_FORBIDDEN_MODEL_COLUMNS = {"record_id", "action_taken", "y_approved"}


class DatasetBSemanticValidationError(ValueError):
    """Raised when the shared Dataset B semantic or split contract is invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class FeatureContractValidation:
    """Validated YAML identity with separate semantic and raw byte digests."""

    semantic_identity: Mapping[str, Any]
    semantic_sha256: str
    raw_sha256: str

    @property
    def model_input_raw_columns(self) -> tuple[str, ...]:
        return tuple(self.semantic_identity["model_input_raw_columns"])

    @property
    def audit_only_columns(self) -> tuple[str, ...]:
        return tuple(self.semantic_identity["audit_only_columns"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_identity": self.semantic_identity,
            "semantic_sha256": self.semantic_sha256,
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True)
class PartitionDataQuality:
    """Observed shape and identity quality for one immutable partition."""

    rows: int
    unique_record_ids: int
    model_input_columns: int
    audit_columns: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "unique_record_ids": self.unique_record_ids,
            "model_input_columns": self.model_input_columns,
            "audit_columns": self.audit_columns,
        }


@dataclass(frozen=True)
class SplitDataQualityValidation:
    """Validation result for the locked split and all three materialized CSVs."""

    split_version: str
    manifest_sha256: str
    feature_contract_semantic_sha256: str
    partitions: Mapping[str, PartitionDataQuality]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "manifest_sha256": self.manifest_sha256,
            "feature_contract_semantic_sha256": self.feature_contract_semantic_sha256,
            "partitions": {
                name: result.as_dict() for name, result in self.partitions.items()
            },
        }


def validate_feature_contract(
    path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
) -> FeatureContractValidation:
    """Validate the locked feature contract and return presentation-stable identity."""
    contract_path = Path(path)
    try:
        raw = contract_path.read_bytes()
        text = raw.decode("utf-8")
        contract = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetBSemanticValidationError(
            f"Cannot read a valid Dataset B feature-contract YAML: {contract_path}"
        ) from exc

    if not isinstance(contract, dict):
        raise DatasetBSemanticValidationError(
            "Dataset B feature-contract YAML root must be a mapping."
        )
    if set(contract) != _EXPECTED_ROOT_KEYS:
        raise DatasetBSemanticValidationError(
            "Dataset B feature-contract root fields differ from the locked schema."
        )
    for field, expected in _EXPECTED_METADATA.items():
        _require_strict_equal(contract.get(field), expected, field)
    _require_strict_equal(
        contract.get("source_of_truth"),
        _EXPECTED_SOURCE_OF_TRUTH,
        "source_of_truth",
    )
    _require_strict_equal(contract.get("target"), _EXPECTED_TARGET, "target")
    _require_strict_equal(
        contract.get("model_input_raw_columns"),
        list(EXPECTED_MODEL_INPUTS),
        "model_input_raw_columns",
    )
    _require_strict_equal(
        contract.get("audit_only_columns"),
        list(EXPECTED_AUDIT_COLUMNS),
        "audit_only_columns",
    )
    _require_strict_equal(
        contract.get("protected_attributes"),
        _EXPECTED_PROTECTED_ATTRIBUTES,
        "protected_attributes",
    )
    _require_strict_equal(
        contract.get("constraints"),
        _EXPECTED_CONSTRAINTS,
        "constraints",
    )
    _validate_column_roles(contract)

    try:
        canonical = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        identity = json.loads(canonical)
    except (TypeError, ValueError) as exc:
        raise DatasetBSemanticValidationError(
            "Dataset B feature contract contains a non-canonical value."
        ) from exc

    return FeatureContractValidation(
        semantic_identity=identity,
        semantic_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_split_data_quality(
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    *,
    expected_partition_rows: Mapping[str, int] = EXPECTED_PARTITION_ROWS,
    expected_manifest_sha256: str | None = LOCKED_SPLIT_MANIFEST_SHA256,
) -> SplitDataQualityValidation:
    """Validate immutable split identity, schema, order, targets, and role separation."""
    contract = validate_feature_contract(feature_contract_path)
    manifest_path = Path(manifest_path)
    try:
        manifest_sha256 = _sha256_file(manifest_path)
    except OSError as exc:
        raise DatasetBSemanticValidationError(
            f"Cannot read Dataset B split manifest: {manifest_path}"
        ) from exc
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256.lower()
    ):
        raise DatasetBSemanticValidationError(
            "Dataset B split manifest differs from the locked split_v1 artifact."
        )
    if set(expected_partition_rows) != {"train", "validation", "test"}:
        raise DatasetBSemanticValidationError(
            "Expected split row counts must cover train, validation, and test."
        )

    try:
        loader = DatasetBSplitLoader(
            manifest_path=manifest_path,
            split_dir=split_dir,
            feature_contract_path=feature_contract_path,
        )
        partitions = {
            name: loader.load(name)  # type: ignore[arg-type]
            for name in ("train", "validation", "test")
        }
    except SplitInputError as exc:
        raise DatasetBSemanticValidationError(
            "Dataset B split data quality is invalid."
        ) from exc

    results: dict[str, PartitionDataQuality] = {}
    all_record_ids: list[str] = []
    for name, partition in partitions.items():
        record_ids = tuple(partition.record_ids.astype(str))
        expected_rows = expected_partition_rows[name]
        if len(record_ids) != expected_rows or len(set(record_ids)) != expected_rows:
            raise DatasetBSemanticValidationError(
                f"Dataset B {name} row count or record-ID uniqueness is invalid."
            )
        if tuple(partition.X.columns) != contract.model_input_raw_columns:
            raise DatasetBSemanticValidationError(
                f"Dataset B {name} model-input order is invalid."
            )
        if tuple(partition.audit.columns) != contract.audit_only_columns:
            raise DatasetBSemanticValidationError(
                f"Dataset B {name} audit-column order is invalid."
            )
        if _FORBIDDEN_MODEL_COLUMNS & set(partition.X.columns):
            raise DatasetBSemanticValidationError(
                f"Dataset B {name} leaks identifier or target data into model inputs."
            )
        if set(partition.X.columns) & set(partition.audit.columns):
            raise DatasetBSemanticValidationError(
                f"Dataset B {name} model and audit columns overlap."
            )
        all_record_ids.extend(record_ids)
        results[name] = PartitionDataQuality(
            rows=len(record_ids),
            unique_record_ids=len(set(record_ids)),
            model_input_columns=len(partition.X.columns),
            audit_columns=len(partition.audit.columns),
        )

    if len(set(all_record_ids)) != len(all_record_ids):
        raise DatasetBSemanticValidationError(
            "Dataset B split partitions contain overlapping record IDs."
        )

    return SplitDataQualityValidation(
        split_version="split_v1",
        manifest_sha256=manifest_sha256,
        feature_contract_semantic_sha256=contract.semantic_sha256,
        partitions=results,
    )


def validate_dataset_b_semantics(
    feature_contract_path: str | Path = DEFAULT_FEATURE_CONTRACT_PATH,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
) -> dict[str, Any]:
    """Return the shared semantic and split-quality evidence used by model routes."""
    feature = validate_feature_contract(feature_contract_path)
    split = validate_split_data_quality(
        feature_contract_path=feature_contract_path,
        manifest_path=manifest_path,
        split_dir=split_dir,
    )
    return {"feature_contract": feature.as_dict(), "split_data_quality": split.as_dict()}


def _validate_column_roles(contract: Mapping[str, Any]) -> None:
    model_columns = tuple(contract["model_input_raw_columns"])
    audit_columns = tuple(contract["audit_only_columns"])
    if len(set(model_columns)) != len(model_columns):
        raise DatasetBSemanticValidationError("Model-input columns must be unique.")
    if len(set(audit_columns)) != len(audit_columns):
        raise DatasetBSemanticValidationError("Audit-only columns must be unique.")
    if set(model_columns) & set(audit_columns):
        raise DatasetBSemanticValidationError("Model and audit columns must not overlap.")
    if _FORBIDDEN_MODEL_COLUMNS & set(model_columns):
        raise DatasetBSemanticValidationError(
            "Identifier or target fields must not enter model inputs."
        )

    protected = contract["protected_attributes"]
    referenced = {protected["sex"]["source_column"]}
    referenced.update(protected["applicant_race"]["source_columns"])
    if not referenced <= set(audit_columns):
        raise DatasetBSemanticValidationError(
            "Protected attributes must reference audit-only columns."
        )


def _require_strict_equal(actual: Any, expected: Any, field: str) -> None:
    if not _strict_equal(actual, expected):
        raise DatasetBSemanticValidationError(
            f"Dataset B feature-contract field {field!r} differs from the locked contract."
        )


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
