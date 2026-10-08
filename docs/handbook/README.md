# Dataset B Handbook

## Purpose

This handbook routes agents and developers to the Dataset B contract,
procedure, and execution-status documents. It is the authoritative entry point
for Dataset B Markdown documentation. [STATUS](STATUS.md) is the sole current
execution-status register. Read only the section needed for the task. The
three instructor-report DOCX files retained under `docs/` remain reports
outside this Markdown handbook.

## Reading path

| If you need to… | Read |
|---|---|
| Understand the dataset, cohort, target, column roles, exclusions, or leakage boundary | [Data contract](contracts/data_contract.md) |
| Validate final model inputs, blank-row handling, or raw `1111` to semantic `Exempt` normalization | [Data-quality gate in the data contract](contracts/data_contract.md#105-data-quality-semantics) |
| Understand feature evidence, the final 15-feature contract, or the final 12-player explanation mapping | [Feature governance](contracts/feature_governance.md) |
| Understand classical preprocessing | [Classical preprocessing contract](contracts/preprocessing_classical.md) |
| Understand neural preprocessing and model-family input routes | [Neural preprocessing contract](contracts/preprocessing_neural.md) |
| Train or validate classical baselines | [Baseline training runbook](runbooks/baseline_training.md) |
| Run neural pilots/training, publish artifacts, recover, or hand off | [Neural training runbook](runbooks/neural_training.md) |
| Implement or execute the locked TabMmini v1 route | [TabMmini v1 training runbook](runbooks/tabmmini_training.md) |
| Understand split, CV, refit, Validation/Test, metrics, bootstrap, or RQ gates | [Split and evaluation contract](contracts/split_and_evaluation.md) |
| Implement or review the locked six-model GSV/Shapley--Lorenz protocol | [RQ2 explanation contract](contracts/rq2_explanation.md) |
| Check current execution stage and evidence | [Current execution status](STATUS.md) |

## Document ownership

- [Data contract](contracts/data_contract.md) owns the stable data boundary:
  cohort, target mapping, column reconciliation, audit roles, exclusions,
  `record_id`, leakage boundary, final input-quality gate, and field-specific
  raw `1111` to semantic `Exempt` normalization.
- [Feature governance](contracts/feature_governance.md) owns screening
  evidence, decision rules, the completed ordered 15-feature contract, and the
  final 12-player explanation mapping.
- [Split and evaluation](contracts/split_and_evaluation.md) owns immutable
  split membership, Train-only CV, full-Train refit, Validation thresholding,
  Test evaluation, metric/comparison boundaries, and RQ2/RQ3 gates.
- [RQ2 explanation](contracts/rq2_explanation.md) owns the six-model RQ2
  player, sample, coalition, explanation, ranking, bootstrap, and acceptance
  protocol. It does not record execution status.
- [Classical preprocessing](contracts/preprocessing_classical.md) owns
  classical parser, transformations, model-family preprocessing, and output
  schema contracts.
- [Neural preprocessing](contracts/preprocessing_neural.md) owns shared
  semantic preprocessing, neural input routes, dtype/container boundaries,
  category support, and preprocessing artifacts.
- [Baseline training](runbooks/baseline_training.md) owns concise execution
  and verification steps for classical baselines.
- [Neural training](runbooks/neural_training.md) owns runtime gates, pilot and
  training procedures, history, publication, recovery, and handoff.
- [TabMmini training](runbooks/tabmmini_training.md) owns the locked TabMmini
  v1 backend, architecture, objective, ensemble, pilot, runtime, prediction,
  recovery, and model-specific implementation policy. It does not record
  execution status.
- [STATUS](STATUS.md) is the sole source for current execution status. Other
  files may define contracts/procedures or retain clearly labeled history, but
  must not act as a parallel current-status register.

## Final retraining policy

The feature and preprocessing documentation is updated in place; this work
does not create a parallel documentation or preprocessing version. Final runs
use directly reviewable schema, configuration, row-alignment, normalization,
prediction, threshold, and metric checks; no opaque content identity is an
acceptance gate.

Each final model execution uses one unique `run_id` to link regenerated
preprocessing, the fresh fitted model, Validation predictions and threshold,
Test predictions, and evaluation outputs. Existing 12-feature fitted
preprocessors, model states, matrices, predictions, thresholds, and evaluation
packages remain historical experimental evidence only. They must not be
resumed, extended, relabeled, or reused by the final 15-feature retraining.

The six final 15-feature reporting inputs are complete and linked through their
`run_id` manifests. The Dataset B RQ2 protocol is locked and its implementation
is pending; RQ3 remains closed until its separately governed grouping,
denominator, and uncertainty decisions are approved. See
[Current execution status](STATUS.md) for the active gates.

## Archive and reports

Files under [archive](archive/) are historical or superseded references. They
are not authoritative and are not part of the default reading path.

The three DOCX files in `docs/` are reports for the instructor. This handbook
does not replace, rename, or supersede those reports.

## Dataset thực nghiệm

Các mô hình hiện tại sử dụng file HMDA New York 2024 tải từ Data Browser trong
khoảng 07–15/09/2026. File này được khóa làm nguồn dữ liệu thực nghiệm; bản tải
ngày 01/10/2026 chỉ dùng đối chiếu và không thay thế dataset huấn luyện.
