"""Targeted synthetic tests for the locked Dataset B TabMmini v1 source."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.dataset_b.models.tabmmini import (
    TabMMiniConfigurationError,
    TabMMiniModel,
    compute_pos_weight,
    expand_member_targets,
    load_tabmmini_v1_config,
    mean_member_probabilities,
    member_wise_weighted_bce,
)
from src.dataset_b.preprocessing import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_SCHEMA,
    TabMMiniCategorySupportError,
    fixed_tabmmini_category_maps,
    make_tabmmini_native_preprocessing_pipeline,
    tabmmini_category_cardinalities,
    validate_tabmmini_category_support,
)
from src.dataset_b.preprocessing.numeric_two_channel import TWO_CHANNEL_COLUMNS
from src.dataset_b.preprocessing.raw_parser import FEATURE_COLUMNS


def _row(**changes):
    row = {
        "loan_type": "1",
        "loan_purpose": "1",
        "lien_status": "1",
        "reverse_mortgage": "2",
        "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2",
        "income": 100.0,
        "debt_to_income_ratio": "42",
        "loan_amount": 200.0,
        "loan_to_value_ratio": 75.0,
        "loan_term": 360.0,
        "occupancy_type": "1",
        "preapproval": "2",
        "interest_only_payment": "2",
        "balloon_payment": "2",
    }
    row.update(changes)
    return row


def _frame(*rows):
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)


def _semantic(*rows):
    pipeline = make_tabmmini_native_preprocessing_pipeline()
    pipeline.fit(_frame(*rows))
    channels = pipeline.named_steps["two_channel"].transform(_frame(*rows))
    return pipeline.named_steps["log1p_numeric"].transform(channels)


class _SyntheticTabM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.make_kwargs = kwargs
        self.bias = nn.Parameter(torch.linspace(-0.5, 0.5, 32).view(1, 32, 1))

    def forward(self, x_num, x_cat):
        base = x_num.mean(dim=1, keepdim=True).view(-1, 1, 1)
        return base.expand(-1, 32, 1) + self.bias


def _synthetic_factory(**kwargs):
    return _SyntheticTabM(**kwargs)


def test_locked_config_schema_and_tabm_make_arguments():
    config = load_tabmmini_v1_config()
    model = TabMMiniModel(config=config, backend_factory=_synthetic_factory)

    assert config["backend"] == {
        "package": "tabm",
        "package_version": "0.0.3",
        "constructor": "tabm.TabM.make",
    }
    assert config["model_parameters"]["num_embeddings"] is None
    assert model.backend.make_kwargs == {
        "n_num_features": 4,
        "cat_cardinalities": list(tabmmini_category_cardinalities()),
        "arch_type": "tabm-mini",
        "k": 32,
        "n_blocks": 2,
        "d_block": 512,
        "dropout": 0.1,
        "d_out": 1,
        "num_embeddings": None,
    }
    broken = copy.deepcopy(config)
    broken["model_parameters"]["k"] = 16
    with pytest.raises(TabMMiniConfigurationError, match="model_parameters"):
        TabMMiniModel(config=broken, backend_factory=_synthetic_factory)


def test_synthetic_normalized_15_to_19_to_native_tensors():
    raw = _frame(
        _row(income=0.0, loan_amount=99.0),
        _row(
            income="NA",
            loan_amount=199.0,
            loan_to_value_ratio="Exempt",
            reverse_mortgage="Exempt",
        ),
    )
    pipeline = make_tabmmini_native_preprocessing_pipeline()
    native = pipeline.fit_transform(raw)

    assert tuple(pipeline.named_steps["two_channel"].get_feature_names_out()) == TWO_CHANNEL_COLUMNS
    assert native.x_num.shape == (2, 4)
    assert native.x_cat.shape == (2, 15)
    assert native.x_num.dtype == torch.float32
    assert native.x_cat.dtype == torch.int64
    assert torch.isfinite(native.x_num).all()
    assert pipeline.named_steps["native"].last_audit_["column_order"] == list(
        TWO_CHANNEL_COLUMNS
    )
    np.testing.assert_allclose(native.x_num[0, :2].numpy(), [np.log1p(0.0), np.log1p(99.0)])


def test_category_maps_are_schema_defined_and_unknowns_fail():
    maps = fixed_tabmmini_category_maps()
    assert tuple(maps) == CATEGORICAL_COLUMNS
    for column in CATEGORICAL_COLUMNS:
        assert maps[column] == {
            category: index for index, category in enumerate(CATEGORICAL_SCHEMA[column])
        }
        assert not ({"1111", "MISSING", "<UNK>"} & set(maps[column]))
    assert "Exempt" in maps["reverse_mortgage"]

    fit = _semantic(_row(), _row(loan_type="2"))
    unknown = fit.copy(deep=True)
    unknown.loc[0, "loan_type"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown.*loan_type"):
        validate_tabmmini_category_support(fit, unknown)


def test_category_support_audit_warns_absent_both_and_fails_absent_fit_present_held_out():
    fit = _semantic(_row(), _row(loan_type="2"))
    same = _semantic(_row())
    report = validate_tabmmini_category_support(fit, same)
    assert report["loan_type"]["3"]["outcome"] == "WARN_ABSENT_BOTH"
    assert report["loan_type"]["1"]["outcome"] == "PASS_SUPPORTED_IN_FIT"

    held_out = _semantic(_row(loan_type="3"))
    with pytest.raises(TabMMiniCategorySupportError, match="loan_type.*'3'"):
        validate_tabmmini_category_support(fit, held_out)


def test_member_wise_weighted_bce_shapes_numerics_and_two_class_guard():
    logits = torch.linspace(-3.0, 4.0, 64, dtype=torch.float32).reshape(2, 32, 1)
    targets = torch.tensor([0.0, 1.0])
    pos_weight = compute_pos_weight(torch.tensor([0, 0, 1]))
    expanded = expand_member_targets(targets, logits)
    loss = member_wise_weighted_bce(logits, targets, pos_weight)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        expanded,
        pos_weight=pos_weight.reshape(1),
        reduction="none",
    ).mean()

    assert expanded.shape == logits.shape == (2, 32, 1)
    assert loss.shape == torch.Size([])
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match="both classes"):
        compute_pos_weight(torch.ones(4))
    with pytest.raises(ValueError, match="exact shape"):
        member_wise_weighted_bce(logits[:, :16], targets, pos_weight)


def test_member_loss_is_not_loss_on_mean_logit():
    logits = torch.cat(
        (torch.full((1, 16, 1), -6.0), torch.full((1, 16, 1), 2.0)),
        dim=1,
    )
    targets = torch.tensor([1.0])
    member_loss = member_wise_weighted_bce(logits, targets, 2.0)
    mean_logit_loss = F.binary_cross_entropy_with_logits(
        logits.mean(dim=1),
        targets.view(-1, 1),
        pos_weight=torch.tensor([2.0]),
    )
    assert not torch.isclose(member_loss, mean_logit_loss)


def test_inference_uses_mean_member_probabilities_not_sigmoid_of_mean_logit():
    logits = torch.cat(
        (torch.full((1, 31, 1), -1.0), torch.full((1, 1, 1), 8.0)),
        dim=1,
    )
    actual = mean_member_probabilities(logits)
    expected = torch.sigmoid(logits).squeeze(-1).mean(dim=1)
    forbidden = torch.sigmoid(logits.mean(dim=1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, forbidden).all()


def test_synthetic_save_load_inference_bundle_round_trip(tmp_path):
    config = load_tabmmini_v1_config()
    raw = _frame(_row(), _row(income=120.0, loan_type="2"))
    native = make_tabmmini_native_preprocessing_pipeline().fit_transform(raw)
    model = TabMMiniModel(config=config, backend_factory=_synthetic_factory).eval()
    with torch.inference_mode():
        before = model.predict_approved_probability(native.x_num, native.x_cat)

    bundle_path = tmp_path / "tabmmini_bundle.pt"
    model.save_inference_bundle(
        bundle_path,
        preprocessing_state={"medians": {"income": 100.0}},
        effective_batch_size=128,
        fixed_epoch=17,
        run_id="synthetic_tabmmini_test",
        runtime_versions={"tabm": "0.0.3", "torch": torch.__version__},
    )
    loaded, payload = TabMMiniModel.load_inference_bundle(
        bundle_path,
        backend_factory=_synthetic_factory,
    )
    with torch.inference_mode():
        after = loaded.predict_approved_probability(native.x_num, native.x_cat)

    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert payload["category_maps"] == fixed_tabmmini_category_maps()
    assert payload["semantic_schema"] == list(TWO_CHANNEL_COLUMNS)
    assert payload["effective_batch_size"] == 128
    assert payload["fixed_epoch"] == 17
    assert "optimizer_state_dict" not in payload
