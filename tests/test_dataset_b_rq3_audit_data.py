from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_b.fairness.audit_data import (
    RQ3InputError,
    load_protocol,
    run_preflight,
    source_ordinals,
)


def test_source_ordinals_are_deterministic_and_strict() -> None:
    observed = source_ordinals(pd.Series(["dataset_b-000000001", "dataset_b-000264298"]))
    np.testing.assert_array_equal(observed, [0, 264297])
    with pytest.raises(RQ3InputError, match="Invalid Dataset B record_id"):
        source_ordinals(pd.Series(["row-1"]))
    with pytest.raises(RQ3InputError, match="positive source-row number"):
        source_ordinals(pd.Series(["dataset_b-000000000"]))


def test_current_frozen_inputs_pass_rq3_preflight() -> None:
    protocol = load_protocol()
    result = run_preflight(protocol)
    assert tuple(result.audit_metadata.columns) == (
        "record_id",
        "y_approved",
        "applicant_sex_code",
        "applicant_sex_group",
        "derived_race",
    )
    assert len(result.audit_metadata) == 43996
    assert result.audit_metadata["record_id"].is_unique
    assert not any(column.startswith("applicant_race-") for column in result.audit_metadata)
    assert list(result.models) == [
        "lr", "rf", "xgboost", "plain_mlp", "realmlp", "tabmmini"
    ]
    for model in result.models.values():
        assert len(model.predictions) == 43996
        assert model.predictions["record_id"].tolist() == result.audit_metadata["record_id"].tolist()
        assert np.isfinite(model.predictions["p_approved"]).all()
        assert model.predictions["p_approved"].between(0.0, 1.0).all()
        assert set(model.predictions["predicted_approved"].unique()) <= {0, 1}
    assert result.bootstrap_indices.shape == (1000, 43996)
    assert result.bootstrap_indices.dtype == np.int32


def test_preflight_code_contains_no_model_execution_calls() -> None:
    source = Path("src/dataset_b/fairness/audit_data.py").read_text(encoding="utf-8")
    for forbidden in ("predict_proba(", ".predict(", ".fit(", "np.random", "default_rng"):
        assert forbidden not in source
