from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.mitigation.inputs import (
    LOCKED_G3,
    LOCKED_G6,
    RQ4AInputError,
    allowed_race_groups,
    load_bootstrap_indices,
    load_config,
    load_frozen_predictions,
    read_frozen_threshold,
    reconcile_audit_metadata,
    sex_mapping,
    split_view_path,
    validate_bootstrap_indices,
    validate_config,
    validate_frozen_predictions,
    validate_run_id,
)
from src.dataset_b.fairness.audit_data import load_protocol as load_rq3_protocol


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def contract(config):
    return validate_config(config)


@pytest.fixture(scope="module")
def rq3_protocol() -> dict:
    return load_rq3_protocol()


def test_locked_config_validates_with_exact_g3_g6_rosters(contract) -> None:
    assert contract.g3 == LOCKED_G3
    assert contract.g6 == LOCKED_G6
    assert contract.configurations["RQ4a-3"].roster == LOCKED_G3
    assert contract.configurations["RQ4a-6"].roster == LOCKED_G6
    assert (contract.configurations["RQ4a-3"].target_eod, contract.configurations["RQ4a-3"].audit_eod) == ("eod3", "eod6")
    assert (contract.configurations["RQ4a-6"].target_eod, contract.configurations["RQ4a-6"].audit_eod) == ("eod6", "eod3")
    assert contract.sensitivity_seeds == tuple(range(42, 72))
    assert (contract.replicates, contract.minimum_valid_replicates) == (1000, 950)
    assert set(contract.baseline_models) == {"xgboost", "realmlp"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["groups"].__setitem__("g6", c["groups"]["g6"][:-1]),
        lambda c: c["groups"].__setitem__("g3", list(reversed(c["groups"]["g3"]))),
        lambda c: c["groups"].__setitem__("g3", c["groups"]["g3"] + ["Joint"]),
        lambda c: c["groups"].__setitem__("allow_drop_merge_rename_or_dynamic_shrink", True),
        lambda c: c["postprocessor"].__setitem__("flip", True),
        lambda c: c["postprocessor"].__setitem__("constraints", "demographic_parity"),
        lambda c: c["randomized_decisions"].__setitem__("sensitivity_seeds", list(range(42, 71))),
        lambda c: c["bootstrap"]["confidence_interval"].__setitem__("quantile_method", "nearest"),
        lambda c: c["configurations"]["RQ4a-3"].__setitem__("target_eod", "eod6"),
        lambda c: c["denominators"].__setitem__("minimum_actual_denied", 10),
        lambda c: c["bootstrap"].__setitem__("test_indices_dtype", "int64"),
        lambda c: c["cached_probability_adapter"].__setitem__("repeated_query_record_id_from_bootstrap", "fail"),
        lambda c: c["cached_probability_adapter"].__setitem__("duplicate_lookup_record_id", "allowed"),
        lambda c: c["audit_join"].pop("controlled_split_views"),
        lambda c: c["audit_join"].__setitem__("derive_split_view_paths_from_test_view", True),
        lambda c: c["baseline_lineage"].__setitem__("rule", "p_approved_gt_frozen_threshold"),
    ],
)
def test_tampered_config_fails_closed(config, mutate) -> None:
    tampered = copy.deepcopy(config)
    mutate(tampered)
    with pytest.raises(RQ4AInputError):
        validate_config(tampered)


def test_run_id_patterns(contract) -> None:
    rq4a3, rq4a6 = contract.configurations["RQ4a-3"], contract.configurations["RQ4a-6"]
    assert validate_run_id("rq4a3_postproc_20261017_01", rq4a3) == "rq4a3_postproc_20261017_01"
    assert validate_run_id("rq4a6_postproc_20261017_12", rq4a6)
    for bad, spec in [
        ("rq4a6_postproc_20261017_01", rq4a3),
        ("rq4a3_postproc_20261317_01", rq4a3),
        ("rq4a3_postproc_20261017_00", rq4a3),
        ("rq4a3_postproc_20261017_1", rq4a3),
        ("rq4a3_postproc_20261017_01_x", rq4a3),
    ]:
        with pytest.raises(RQ4AInputError):
            validate_run_id(bad, spec)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {"record_id": ["r1", "r2", "r3"], "y_approved": [1, 0, 1], "p_approved": [0.9, 0.2, 0.55]}
    )


def test_frozen_predictions_accept_matching_order_and_labels() -> None:
    out = validate_frozen_predictions(
        _predictions(), model="xgboost", partition="test", expected_record_ids=["r1", "r2", "r3"], expected_y=[1, 0, 1],
        threshold=0.55,
    )
    assert out["record_id"].tolist() == ["r1", "r2", "r3"]
    assert out["p_approved"].dtype == np.float64
    # Validation artifact without a stored hard class derives p >= threshold (0.55 >= 0.55 is Approved).
    assert out["predicted_approved"].tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    "frame_change, ids, labels",
    [
        (None, ["r2", "r1", "r3"], [0, 1, 1]),
        (None, ["r1", "r2"], [1, 0]),
        (None, ["r1", "r2", "r4"], [1, 0, 1]),
        (None, ["r1", "r2", "r3"], [1, 1, 1]),
        (lambda f: f.assign(record_id=["r1", "r1", "r3"]), ["r1", "r1", "r3"], [1, 0, 1]),
        (lambda f: f.assign(y_approved=[1, 2, 1]), ["r1", "r2", "r3"], [1, 0, 1]),
        (lambda f: f.drop(columns="y_approved"), ["r1", "r2", "r3"], [1, 0, 1]),
    ],
)
def test_id_order_and_label_mismatch_fail_closed(frame_change, ids, labels) -> None:
    frame = _predictions() if frame_change is None else frame_change(_predictions())
    with pytest.raises(RQ4AInputError):
        validate_frozen_predictions(frame, model="realmlp", partition="validation", expected_record_ids=ids, expected_y=labels,
                                    threshold=0.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01, "x"])
def test_invalid_probabilities_fail_closed(bad) -> None:
    frame = _predictions().astype({"p_approved": object})
    frame.loc[1, "p_approved"] = bad
    with pytest.raises(RQ4AInputError):
        validate_frozen_predictions(frame, model="xgboost", partition="test", expected_record_ids=["r1", "r2", "r3"], expected_y=[1, 0, 1],
                                    threshold=0.5)


def test_csv_loading_uses_round_trip_float_precision(tmp_path) -> None:
    rng = np.random.default_rng(7)
    p = rng.random(500)
    frame = pd.DataFrame({"record_id": [f"r{i}" for i in range(500)], "y_approved": rng.integers(0, 2, 500), "p_approved": p})
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    out = load_frozen_predictions(
        path, model="xgboost", partition="test", expected_record_ids=frame["record_id"].tolist(), expected_y=frame["y_approved"],
        threshold=0.5,
    )
    assert np.array_equal(out["p_approved"].to_numpy(), p)
    assert np.array_equal(out["predicted_approved"].to_numpy(), (p >= 0.5).astype(np.int8))


def test_threshold_artifact_must_match_run_and_frozen_value(tmp_path, contract) -> None:
    spec = contract.baseline_models["xgboost"]
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"run_id": spec.run_id, "selected_threshold": spec.frozen_threshold}), encoding="utf-8")
    assert read_frozen_threshold(path, spec) == spec.frozen_threshold
    path.write_text(json.dumps({"run_id": "other", "selected_threshold": spec.frozen_threshold}), encoding="utf-8")
    with pytest.raises(RQ4AInputError):
        read_frozen_threshold(path, spec)
    path.write_text(json.dumps({"run_id": spec.run_id, "selected_threshold": 0.5}), encoding="utf-8")
    with pytest.raises(RQ4AInputError):
        read_frozen_threshold(path, spec)


def _audit_toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(
        {
            "lei": ["L1", "L2", "L3", "L4"],
            "action_taken": ["1", "3", "1", "3"],
            "applicant_sex": ["1", "2", "3", "1"],
            "derived_race": ["White", "Asian", "Joint", "Race Not Available"],
        }
    )
    view = pd.DataFrame(
        {
            "record_id": ["dataset_b-000000003", "dataset_b-000000001", "dataset_b-000000004"],
            "lei": ["L3", "L1", "L4"],
            "action_taken": ["1", "1", "3"],
            "y_approved": ["1", "1", "0"],
            "applicant_sex": ["3", "1", "1"],
        }
    )
    return view, raw


def test_audit_metadata_generalizes_to_validation_and_test(rq3_protocol) -> None:
    view, raw = _audit_toy()
    for partition in ("validation", "test"):
        out = reconcile_audit_metadata(
            view, raw, partition=partition, expected_rows=3,
            sex_groups=sex_mapping(rq3_protocol), race_groups=allowed_race_groups(rq3_protocol),
        )
        assert out["record_id"].tolist() == view["record_id"].tolist()
        assert out["derived_race"].tolist() == ["Joint", "White", "Race Not Available"]
        assert out["applicant_sex_group"].tolist()[1] == "Male"
        assert out["y_approved"].tolist() == [1, 1, 0]


@pytest.mark.parametrize(
    "mutate_view, mutate_raw, partition, rows",
    [
        (lambda v: v.assign(lei=["L3", "LX", "L4"]), None, "validation", 3),
        (lambda v: v.assign(action_taken=["1", "3", "3"]), None, "test", 3),
        (lambda v: v.assign(applicant_sex=["3", "9", "1"]), None, "test", 3),
        (None, lambda r: r.assign(derived_race=["Martian", "Asian", "Joint", "x"]), "test", 3),
        (lambda v: v.assign(record_id=["dataset_b-000000003", "bad", "dataset_b-000000004"]), None, "test", 3),
        (lambda v: v.assign(record_id=["dataset_b-000000003", "dataset_b-000000009", "dataset_b-000000004"]), None, "test", 3),
        (None, None, "test", 4),
        (None, None, "train", 3),
    ],
)
def test_audit_join_fails_closed(rq3_protocol, mutate_view, mutate_raw, partition, rows) -> None:
    view, raw = _audit_toy()
    view = view if mutate_view is None else mutate_view(view)
    raw = raw if mutate_raw is None else mutate_raw(raw)
    with pytest.raises(RQ4AInputError):
        reconcile_audit_metadata(
            view, raw, partition=partition, expected_rows=rows,
            sex_groups=sex_mapping(rq3_protocol), race_groups=allowed_race_groups(rq3_protocol),
        )


def test_split_view_paths_are_explicit_in_config(contract) -> None:
    assert split_view_path("validation", contract).as_posix().endswith("data/dataset_b/split_v1/validation.csv")
    assert split_view_path("test", contract).as_posix().endswith("data/dataset_b/split_v1/test.csv")
    with pytest.raises(RQ4AInputError):
        split_view_path("train", contract)


def test_bootstrap_indices_validation(tmp_path) -> None:
    good = np.zeros((3, 5), dtype=np.int32)
    assert validate_bootstrap_indices(good, replicates=3, test_rows=5) is good
    for bad in (np.zeros((2, 5), dtype=np.int32), np.full((3, 5), 5, dtype=np.int32), np.zeros((3, 5)),
                np.zeros((3, 5), dtype=np.int64)):
        with pytest.raises(RQ4AInputError):
            validate_bootstrap_indices(bad, replicates=3, test_rows=5)
    path = tmp_path / "indices.npz"
    np.savez(path, indices=good)
    assert load_bootstrap_indices(path, replicates=3, test_rows=5).shape == (3, 5)
    extra = tmp_path / "extra.npz"
    np.savez(extra, indices=good, other=good)
    with pytest.raises(RQ4AInputError):
        load_bootstrap_indices(extra, replicates=3, test_rows=5)


def test_int64_bootstrap_indices_fail(tmp_path) -> None:
    path = tmp_path / "indices64.npz"
    np.savez(path, indices=np.zeros((3, 5), dtype=np.int64))
    with pytest.raises(RQ4AInputError):
        load_bootstrap_indices(path, replicates=3, test_rows=5)


def test_stored_hard_class_lineage() -> None:
    kwargs = dict(model="xgboost", partition="test", expected_record_ids=["r1", "r2", "r3"], expected_y=[1, 0, 1], threshold=0.5)
    good = _predictions().assign(predicted_approved=[1, 0, 1], predicted_denied=[0, 1, 0])
    out = validate_frozen_predictions(good, **kwargs)
    assert out["predicted_approved"].tolist() == [1, 0, 1]
    for tampered in (
        _predictions().assign(predicted_approved=[1, 1, 1]),
        _predictions().assign(predicted_approved=[1, 0, 1], predicted_denied=[0, 1, 1]),
        _predictions().assign(predicted_denied=[1, 0, 1]),
        _predictions().assign(predicted_approved=[1, 0, 2]),
        _predictions().assign(predicted_approved=[1, None, 1]),
    ):
        with pytest.raises(RQ4AInputError):
            validate_frozen_predictions(tampered, **kwargs)


_DELETE = object()

# One mutation per case: (path inside the config, replacement value or _DELETE).
C4_TAMPER_CASES = [
    (("bootstrap", "calibration_resampling", "source"), "train"),
    (("bootstrap", "calibration_resampling", "method"), "bootstrap_with_replacement"),
    (("bootstrap", "calibration_resampling", "strata"), ["y_approved", "derived_race"]),
    (("bootstrap", "calibration_resampling", "strata"), ["derived_race"]),
    (("bootstrap", "calibration_resampling", "refit_postprocessor_each_replicate"), False),
    (("bootstrap", "calibration_resampling", "refit_postprocessor_each_replicate"), 1),
    (("bootstrap", "calibration_resampling"), _DELETE),
    (("bootstrap", "test_indices"), "artifacts/dataset_b/other_indices.npz"),
    (("bootstrap", "reuse_locked_paired_test_indices"), False),
    (("bootstrap", "allow_regenerate_test_indices"), True),
    (("bootstrap", "allow_regenerate_test_indices"), 0),
    (("bootstrap", "root_seed_formula"), "20000 + replicate_id"),
    (("bootstrap", "root_seed_formula"), _DELETE),
    (("bootstrap", "independent_child_streams"), ["randomized_decisions", "calibration_resampling"]),
    (("bootstrap", "independent_child_streams"), ["calibration_resampling"]),
    (("bootstrap", "locked_test_indices_consume_new_rng_stream"), True),
    (("bootstrap", "randomized_draw_order"), "row_order"),
    (("bootstrap", "same_seed_and_order_reproduce_bit_for_bit"), False),
    (("bootstrap", "same_seed_and_order_reproduce_bit_for_bit"), _DELETE),
    (("bootstrap", "test_indices_dtype"), "int64"),
    (("bootstrap", "repeated_test_occurrences_receive_independent_randomized_draws"), False),
    (("cached_probability_adapter", "preserve_query_multiplicity_and_order"), False),
    (("cached_probability_adapter", "preserve_query_multiplicity_and_order"), _DELETE),
    (("cached_probability_adapter", "synthetic_row_keys_allowed"), True),
    (("cached_probability_adapter", "duplicate_lookup_record_id"), "allowed"),
    (("cached_probability_adapter", "on_unknown_or_missing_query_record_id"), "impute"),
    (("cached_probability_adapter", "repeated_query_record_id_from_bootstrap"), "fail"),
    (("baseline_lineage", "rule"), "p_approved_gt_frozen_threshold"),
    (("baseline_lineage", "stored_predicted_approved_must_match_rule"), False),
    (("baseline_lineage", "stored_predicted_denied_must_equal_1_minus_predicted_approved"), False),
    (("baseline_lineage", "validated_frame_includes_predicted_approved"), False),
    (("baseline_lineage", "derive_when_no_stored_hard_class"), False),
    (("baseline_lineage", "derive_when_no_stored_hard_class"), _DELETE),
    (("metrics", "denied_metric_definitions", "positive_class"), "approved"),
    (("metrics", "denied_metric_definitions", "precision_none_when_no_predicted_denied"), False),
    (("metrics", "denied_metric_definitions", "recall_none_when_no_actual_denied"), False),
    (("metrics", "denied_metric_definitions", "f1_formula"), "2 * precision * recall / (precision + recall)"),
    (("metrics", "denied_metric_definitions", "f1_none_only_when_denominator_zero"), False),
    (("metrics", "denied_metric_definitions"), _DELETE),
    (("metrics", "empty_group_change_rate"), 0.0),
    (("metrics", "empty_group_change_rate"), _DELETE),
    (("audit_join", "controlled_split_views", "validation"), "data/dataset_b/split_v2/validation.csv"),
    (("audit_join", "controlled_split_views", "test"), "data/dataset_b/split_v1/validation.csv"),
    (("audit_join", "controlled_split_views"), {"test": "data/dataset_b/split_v1/test.csv",
                                                 "validation": "data/dataset_b/split_v1/validation.csv"}),
    (("audit_join", "derive_split_view_paths_from_test_view"), True),
]


@pytest.mark.parametrize(
    "path, value",
    C4_TAMPER_CASES,
    ids=[".".join(path) + ("=<deleted>" if value is _DELETE else f"={value!r}") for path, value in C4_TAMPER_CASES],
)
def test_each_c4_locked_field_tamper_fails_closed(config, path, value) -> None:
    tampered = copy.deepcopy(config)
    parent = tampered
    for key in path[:-1]:
        parent = parent[key]
    assert path[-1] in parent, f"fixture path {path} is not in the locked config"
    if value is _DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    with pytest.raises(RQ4AInputError):
        validate_config(tampered)
