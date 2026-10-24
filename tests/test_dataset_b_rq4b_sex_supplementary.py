"""Focused safeguards for the RQ4b applicant_sex supplementary package."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset_b.mitigation import rq4b_evaluation as ev
from src.dataset_b.mitigation import rq4b_sex_supplementary as sex


def _pred(offset: float) -> ev.PartitionPredictions:
    n = 500
    ids = tuple(f"id-{i:04d}" for i in range(n))
    y = np.tile(np.array([0, 1], dtype=np.int8), n // 2)
    groups = np.repeat(np.array(ev.SEX_LABELS, dtype=object), n // len(ev.SEX_LABELS))
    p = np.clip(.35 + .35 * y + offset + np.linspace(-.1, .1, n), 0, 1)
    return ev.PartitionPredictions(ids, y, p, np.repeat("White", n), groups)


def _arms():
    rows = []
    for lam, seed in ((0., 42), (0., 43), (0., 44), (.01, 42), (.01, 43), (.01, 44), (.03, 42), (.1, 42), (.3, 42)):
        rows.append(SimpleNamespace(adversary_lambda=lam, seed=seed, role="control" if lam == 0 else "comparator", run_id="source", threshold=.5, artifact_id=f"{lam}-{seed}"))
    return rows


def _data():
    arms = _arms()
    test = {a.artifact_id: _pred(a.adversary_lambda / 10) for a in arms}
    validation = {a.artifact_id: _pred(a.adversary_lambda / 20) for a in arms}
    # deterministic locked-shaped bootstrap payload; table logic must retain every replicate id.
    indices = np.tile(np.arange(500, dtype=np.int32), (1000, 1))
    return arms, test, validation, indices


def test_rq3_mapping_and_fixed_primary_roster_are_locked():
    protocol = {"attributes": {"applicant_sex": {"primary_groups": {"Male": "1", "Female": "2"}, "coverage_only_groups": {"Information not provided": "3", "Not applicable": "4", "Applicant selected both male and female": "6"}}}}
    assert ev.assert_rq3_sex_mapping(protocol) == ev.RQ3_SEX_CODES
    assert sex.SEX_ROSTER == ("Male", "Female")
    with pytest.raises(ev.RQ4bEvaluationError, match="raw RQ3"):
        ev.validate_sex_labels(["1"])


def test_group_schema_includes_all_five_groups_and_matched_pairs():
    tables = sex._tables(*_data())
    groups = tables["sex_group_metrics.csv"]
    assert len(groups) == 9 * 2 * 5
    assert set(row["group"] for row in groups) == set(ev.SEX_LABELS)
    assert len(tables["sex_matched_deltas.csv"]) == 6
    assert len(tables["sex_decision_changes.csv"]) == 6 * 6  # overall + five sex groups


def test_bootstrap_payload_uses_all_1000_replicates_and_fixed_digest_shape():
    tables = sex._tables(*_data())
    rows = tables["sex_bootstrap_replicates.csv"]
    assert len(rows) == 6000
    assert {row["replicate_id"] for row in rows} == set(range(1000))
    assert all(row["sample_size"] == 500 for row in rows)
    assert all(row["valid"] for row in rows)


def test_package_recomputes_and_rejects_forged_output_without_fit_or_inference(tmp_path, monkeypatch):
    arms, test, validation, indices = _data()
    monkeypatch.setattr(sex, "ROOT", tmp_path)
    monkeypatch.setattr(sex, "_verify_sources", lambda: (arms, {"pinned": "digest"}))
    monkeypatch.setattr(sex, "_load_saved", lambda _: (test, validation, indices))
    target = sex.evaluate()
    assert target.exists()
    receipt = sex.validate_package()
    assert receipt["status"] == "PASS"

    # A forged output and matching forged evaluation receipt still fails deterministic recomputation.
    package = target / "stages" / sex.STAGE / "sex_arm_metrics.csv"
    package.write_bytes(package.read_bytes().replace(b"0.0", b"9.9", 1))
    with pytest.raises(sex.SexSupplementaryError, match="digest differs|does not match recomputation"):
        sex.validate_package()


def test_atomic_publish_failure_leaves_no_complete_run(tmp_path, monkeypatch):
    arms, test, validation, indices = _data()
    monkeypatch.setattr(sex, "ROOT", tmp_path)
    monkeypatch.setattr(sex, "_verify_sources", lambda: (arms, {"pinned": "digest"}))
    monkeypatch.setattr(sex, "_load_saved", lambda _: (test, validation, indices))
    real = sex.os.rename
    monkeypatch.setattr(sex.os, "rename", lambda *_: (_ for _ in ()).throw(PermissionError("simulated publish failure")))
    with pytest.raises(PermissionError, match="simulated publish failure"):
        sex.evaluate()
    assert not (tmp_path / sex.RUN_ID).exists()
    assert not list(tmp_path.glob(".tmp_*"))
    monkeypatch.setattr(sex.os, "rename", real)
