"""Contract tests for the RMLP-06A fixed-epoch one-fold smoke."""

import numpy as np

from src.dataset_b.runtime.realmlp_cv_smoke import (
    _archive_interrupted_working_directory,
    _atomic_rename_probe,
    _validate_predictions,
    _write_predictions,
)


def test_atomic_rename_probe_passes_without_leaving_probe_paths(tmp_path):
    result = _atomic_rename_probe(tmp_path)

    assert result == {
        "status": "PASS",
        "operation": "same_filesystem_os_replace_directory",
    }
    assert list(tmp_path.iterdir()) == []


def test_interrupted_working_directory_is_archived_without_overwrite(tmp_path):
    working = tmp_path / "fold_01.tmp"
    working.mkdir()
    (working / "run.log").write_text("interrupted\n", encoding="utf-8")
    first = _archive_interrupted_working_directory(working)

    working.mkdir()
    (working / "run.log").write_text("second\n", encoding="utf-8")
    second = _archive_interrupted_working_directory(working)

    assert first == "fold_01.failed_attempt_001"
    assert second == "fold_01.failed_attempt_002"
    assert (tmp_path / first / "run.log").read_text(encoding="utf-8") == "interrupted\n"
    assert (tmp_path / second / "run.log").read_text(encoding="utf-8") == "second\n"
    assert not working.exists()


def test_prediction_artifact_preserves_exact_membership_and_probabilities(tmp_path):
    path = tmp_path / "predictions.csv"
    record_ids = ("r1", "r2")
    labels = np.asarray([0, 1], dtype=np.int64)
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)

    _write_predictions(path, record_ids, labels, probabilities)
    _validate_predictions(path, record_ids)

    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "record_id,y_true,probability_0,probability_1"
    )
