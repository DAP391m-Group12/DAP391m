"""Focused boundary tests for the direct LR-10 reviewer."""

import pytest

from src.dataset_b.models.classical.lr import review
from src.dataset_b.review import lr_final_review as compatibility


def test_legacy_review_module_delegates_to_canonical():
    assert compatibility.LRFinalReviewError is review.LRFinalReviewError
    assert compatibility.review_lr_final is review.review_lr_final


def test_review_requires_explicit_frozen_run_id(tmp_path):
    root = tmp_path / "other_run"
    root.mkdir()
    with pytest.raises(review.LRFinalReviewError, match="explicit run ID"):
        review.review_lr_final("other_run", run_dir=root)


def test_review_refuses_outside_working_or_final_roots(tmp_path):
    root = tmp_path / review.RUN_ID
    root.mkdir()
    with pytest.raises(review.LRFinalReviewError, match="outside the LR working/final roots"):
        review.review_lr_final(review.RUN_ID, run_dir=root)


def test_review_accepts_final_root_before_direct_validation(tmp_path, monkeypatch):
    final_root = tmp_path / "final"
    root = final_root / review.RUN_ID
    root.mkdir(parents=True)
    monkeypatch.setattr(review, "DEFAULT_LR_ROOT", tmp_path / "working")
    monkeypatch.setattr(review, "DEFAULT_LR_FINAL_ROOT", final_root)
    monkeypatch.setattr(
        review,
        "_validate_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("final root accepted")),
    )
    with pytest.raises(RuntimeError, match="final root accepted"):
        review.review_lr_final(review.RUN_ID, run_dir=root)


def test_direct_review_source_has_no_digest_gate():
    source = review.Path(review.__file__).read_text(encoding="utf-8").lower()
    assert "hashlib" not in source
    assert "sha256" not in source
    assert "predict_proba(" not in source
    assert ".fit(" not in source
