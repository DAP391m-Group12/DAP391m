# Dataset B Source Guide

`src/dataset_b` contains the Dataset B research implementation. The public
handover entry point is intentionally read-only and CPU-safe; importing it does
not import model, checkpoint, CUDA, training, evaluation, or inference code.

## Public handover entry points

| File | Role |
|---|---|
| `core/config.py` | Resolve logical config roles through `config/dataset_b/catalog.yaml` |
| `core/navigation.py` | Static config, frozen-marker, notebook, and report-index validation |
| `cli.py` | Public read-only command line interface |

Examples:

```text
python -m src.dataset_b.cli status
python -m src.dataset_b.cli validate-config
python -m src.dataset_b.cli validate-artifacts --scope frozen
python -m src.dataset_b.cli validate-notebooks --scope final
python -m src.dataset_b.cli report-index
```

These commands do not fit or load models, run inference, select thresholds,
regenerate bootstrap results, execute notebooks, or publish artifacts. There
are deliberately no public `train`, `evaluate`, `predict`, or `serve` commands
in the handover CLI.

Configuration roles are resolved as dotted names:

```python
from src.dataset_b import config_path

features = config_path("contracts.features")
realmlp = config_path("models.realmlp")
rq4b = config_path("research.rq4b")
```

## Current internal ownership

T05A1 moved the shared cohort-view and split-manifest implementations into
`data/`. T05A2A added raw-token parsing and the split-input quality boundary to
the same canonical package. T05A2B added fold-local median imputation, numeric
value/status channels, and fixed-schema categorical encoding. The former `preparation.dataset_view`,
`splits.split_manifest`, `preprocessing.raw_parser`, and
`preprocessing.split_inputs`, `preprocessing.median_imputer`,
`preprocessing.numeric_two_channel`, and `preprocessing.categorical_encoder`
paths remain compatibility-only re-exports for
active consumers. The split creation workflow now lives at
`workflows/create_split.py`; `splits/run_split.py` remains a compatibility and
CLI path.

T05A3 moved the model-independent semantic feature-contract and immutable
split-quality validator into `core/feature_contract.py`. The former
`contracts.feature_contract` path remains a compatibility-only re-export.

T05B1 moved the LR-selective log transformer and LR/RF/XGBoost profile builder
into `models/classical/`. The numeric-value `RobustScaler` shared by LR and
PlainMLP lives in `data/numeric_value_scaler.py`. Former preprocessing paths
remain compatibility-only re-exports. The model package uses lazy estimator
exports, so importing `models` or `models.classical` does not load a neural
backend.

T05B2A moved the LR-08 Validation, LR-09 frozen Test evaluation, and LR-10
read-only final-review implementations into `models/classical/lr/`. Their
former `evaluation/` and `review/` paths remain compatibility delegates. LR-07
was originally executed on 2026-10-02 as an inline script that was never saved
in the repository. Its training procedure was reconstructed afterwards from the
recorded execution session into `models/classical/lr/training.py`; that module
was not used to create the frozen bundle and its `run` function has not been
executed on the real data. LR-07
remains frozen and must not be retrained; use its read-only `validate` command
to check the frozen bundle.

T05B2B moved RF-07 training, RF-08 Validation, RF-09 frozen Test evaluation,
and RF-10 read-only final review into `models/classical/rf/`. Former training,
evaluation, and review paths remain compatibility and CLI delegates.

T05B2C moved XGB-07 training, XGB-08 Validation, XGB-09 frozen Test
evaluation, and XGB-10 read-only final review into
`models/classical/xgboost/`. Former training, evaluation, and review paths
remain compatibility and CLI delegates.

The final T05B phase moved split creation and the classical fold-local CV
export into `workflows/create_split.py` and
`workflows/export_classical_cv.py`. Former split and preprocessing workflow
paths remain compatibility and CLI delegates. Classical ownership is closed;
shared six-model reporting remains deferred to RQ1 ownership.

T05C consolidated shared neural preprocessing under `models/neural/common/`
and the PlainMLP, RealMLP, and TabMmini lifecycles under their respective
`models/neural/` packages. Neural entry points now live in `workflows/`; former
paths remain compatibility delegates and optional backends are lazy-loaded.

T05D moved the shared six-model reporting contract into `research/rq1/`,
GSV/SL into `research/rq2/`, and descriptive fairness auditing into
`research/rq3/`. Former modules remain compatibility aliases; the fairness
aliases are retained for frozen RQ4b imports.

T05E moved the non-frozen RQ4a input, metric, postprocessing, and runner
implementations into `research/rq4/rq4a/`. Former `mitigation/` paths remain
module aliases, including historical module identifiers used by persisted
RQ4a records. The five protected RQ4b modules, frozen RQ4b config,
`rq4b_spike.py`, and `rq4b_sex_supplementary.py` remain in place.

Compatibility modules and protected source remain at historical paths where
active consumers or frozen lineage require them. Do not move or edit a
hash-pinned source merely to make the physical tree uniform.

Checkpoint persistence remains model-specific. This navigation layer does not
create deployment serving, checkpoint migration, or resume support.
