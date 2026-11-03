"""Score new records with one frozen checkpoint; runs in the model's profile. Nothing is fitted."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_b.data.split_inputs import FEATURE_COLUMNS, DatasetBInputQualityGate, DatasetBPartition
from src.utils.path import PROJECT_ROOT

from .manifest import ReproduceError
from .model_worker import PREDICTORS, _cuda_evidence, _read_json, _require_cuda
from .spec import MODELS, check_run_identity


REQUIRED_COLUMNS = ("record_id", *FEATURE_COLUMNS)


def read_input(path: Path) -> pd.DataFrame:
    """Read new records as text, exactly as the raw parser expects them."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ReproduceError(f"Input is missing required columns: {missing}")
    return frame


def gate_input(frame: pd.DataFrame) -> Any:
    """Apply the real input-quality gate; new records have no label, so these fields are placeholders."""
    index = frame.index
    partition = DatasetBPartition(
        name="test",
        X=frame[list(FEATURE_COLUMNS)].copy(),
        y=pd.Series(1, index=index),
        record_ids=frame["record_id"].copy(),
        audit=pd.DataFrame(index=index),
        action_taken=pd.Series("1", index=index),
    )
    return DatasetBInputQualityGate().apply(partition)


def _realmlp_probabilities(checkpoint: Path, gated: Any) -> np.ndarray:
    import joblib

    from src.dataset_b.models.neural.realmlp.final_contract import transform_native_preprocessing, validate_probabilities

    bundle = joblib.load(checkpoint)
    prepared = transform_native_preprocessing(bundle["preprocessing"], gated)
    return validate_probabilities(bundle["estimator"].predict_proba(prepared.X), len(prepared.X))[:, 1]


def predict(key: str, input_path: Path, output_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    spec = MODELS[key]
    run_root = project_root / spec.run_root
    selection = _read_json(run_root / spec.threshold_file)
    check_run_identity(spec, selection)
    threshold = float(selection["selected_threshold"])
    gated = gate_input(read_input(input_path))
    checkpoint = run_root / spec.checkpoint
    _require_cuda(spec)
    if key == "realmlp":
        probabilities = _realmlp_probabilities(checkpoint, gated)
    else:
        probabilities = PREDICTORS[key](checkpoint, gated.X.reset_index(drop=True))
    device = _cuda_evidence(spec)
    if probabilities.shape != (len(gated.record_ids),) or not np.isfinite(probabilities).all():
        raise ReproduceError(f"{key}: probabilities are invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id", "p_approved", "threshold", "predicted_approved"))
        for record_id, probability in zip(gated.record_ids.astype(str), probabilities, strict=True):
            writer.writerow((record_id, repr(float(probability)), threshold, int(probability >= threshold)))
    return {"status": "PASS", "model": key, "run_id": spec.run_id, "rows": len(probabilities), "threshold": threshold, "compute": device}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.input, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
