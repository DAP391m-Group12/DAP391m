"""Compatibility tests for the PMLP-F04 full-Train facade."""

from src.dataset_b.runtime import plain_mlp_full_train as facade
from src.dataset_b.runtime import plain_mlp_training as implementation


def test_full_train_facade_binds_only_to_f04_implementation():
    assert facade.run_plain_mlp_full_train is implementation.run_plain_mlp_full_train
    assert (
        facade.validate_plain_mlp_full_train
        is implementation.validate_plain_mlp_full_train
    )
    assert facade.load_plain_mlp_bundle is implementation.load_plain_mlp_bundle
    assert facade.PlainMLPFullTrainError is implementation.PlainMLPFullTrainError
    assert facade.EXPECTED_FEATURES == 62
    assert facade.EXPECTED_TRAIN_ROWS == 205_310


def test_full_train_facade_preserves_semantic_lineage_policy():
    expected = {
        "split_manifest_sha256": "a" * 64,
        "feature_contract_sha256": "b" * 64,
    }
    presentation_only_drift = {
        "split_manifest_sha256": "a" * 64,
        "feature_contract_sha256": "c" * 64,
    }
    semantic_drift = {
        "split_manifest_sha256": "d" * 64,
        "feature_contract_sha256": "c" * 64,
    }

    assert facade._lineage_matches_feature_provenance(
        presentation_only_drift, expected
    )
    assert not facade._lineage_matches_feature_provenance(semantic_drift, expected)
