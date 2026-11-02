from src.dataset_b.data import DatasetBSplitter, DatasetBView, SplitManifestError
from src.dataset_b.preparation import DatasetBView as LegacyDatasetBView
from src.dataset_b.splits import DatasetBSplitter as LegacyDatasetBSplitter
from src.dataset_b.splits import run_split as compatibility_workflow
from src.dataset_b.workflows import create_split


def test_legacy_split_imports_are_canonical():
    assert LegacyDatasetBView is DatasetBView
    assert LegacyDatasetBSplitter is DatasetBSplitter


def test_legacy_split_workflow_delegates_to_canonical():
    assert compatibility_workflow.SplitDataExportError is create_split.SplitDataExportError
    assert compatibility_workflow.main is create_split.main


def _contract() -> dict:
    return {
        "dataset": "dataset_b",
        "split_version": "split_v1",
        "split_unit": "record_id",
        "grouping_constraint": "none",
        "random_state": 42,
        "holdout_split": {
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "stratify_by": "y_approved",
        },
        "cross_validation": {
            "splitter": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "allowed_population": "train_only",
        },
    }


def _view(total_records=100) -> DatasetBView:
    records = tuple(
        {"record_id": f"record-{index:03d}", "y_approved": index % 2}
        for index in range(total_records)
    )
    return DatasetBView(records=records, model_input_columns=(), audit_only_columns=())


def test_manifest_is_deterministic_disjoint_and_complete():
    splitter = DatasetBSplitter(_contract())

    first = splitter.build(_view())
    second = splitter.build(_view())

    assert first == second
    assert len(first.train_record_ids) == 70
    assert len(first.validation_record_ids) == 15
    assert len(first.test_record_ids) == 15
    assert len(set(first.all_record_ids)) == 100


def test_cv_uses_train_population_only():
    splitter = DatasetBSplitter(_contract())
    manifest = splitter.build(_view())
    train_labels = [int(record_id.rsplit("-", 1)[1]) % 2 for record_id in manifest.train_record_ids]

    folds = list(splitter.cv_splitter().split(manifest.train_record_ids, train_labels))

    assert len(folds) == 5
    assert all(set(test_index).issubset(set(range(len(manifest.train_record_ids)))) for _, test_index in folds)
    assert all(len(test_index) == 14 for _, test_index in folds)


def test_manifest_rejects_duplicate_record_ids():
    duplicate_view = DatasetBView(
        records=(
            {"record_id": "duplicate", "y_approved": 0},
            {"record_id": "duplicate", "y_approved": 1},
        ),
        model_input_columns=(),
        audit_only_columns=(),
    )

    try:
        DatasetBSplitter(_contract()).build(duplicate_view)
    except SplitManifestError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Duplicate record IDs must be rejected.")


def test_manifest_persistence_round_trip(tmp_path):
    splitter = DatasetBSplitter(_contract())
    manifest = splitter.build(_view())
    output_path = splitter.write_manifest(manifest, tmp_path / "split_v1.json")

    assert output_path.exists()
    assert DatasetBSplitter.load_manifest(output_path) == manifest
