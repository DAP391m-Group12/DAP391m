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
| Understand feature evidence, decision rules, temporary locks, or open decisions | [Feature governance](contracts/feature_governance.md) |
| Understand classical preprocessing | [Classical preprocessing contract](contracts/preprocessing_classical.md) |
| Understand neural preprocessing and model-family input routes | [Neural preprocessing contract](contracts/preprocessing_neural.md) |
| Train or validate classical baselines | [Baseline training runbook](runbooks/baseline_training.md) |
| Run neural pilots/training, publish artifacts, recover, or hand off | [Neural training runbook](runbooks/neural_training.md) |
| Understand split, CV, refit, Validation/Test, metrics, bootstrap, or RQ gates | [Split and evaluation contract](contracts/split_and_evaluation.md) |
| Check current execution stage and evidence | [Current execution status](STATUS.md) |

## Document ownership

- [Data contract](contracts/data_contract.md) owns the stable data boundary:
  cohort, target mapping, column reconciliation, audit roles, exclusions,
  `record_id`, leakage boundary, and schema gates.
- [Feature governance](contracts/feature_governance.md) owns screening
  evidence, decision rules, provisional locks, representation alternatives,
  and conditions for a final feature lock.
- [Split and evaluation](contracts/split_and_evaluation.md) owns immutable
  split membership, Train-only CV, full-Train refit, Validation thresholding,
  Test evaluation, metric/comparison boundaries, and RQ2/RQ3 gates.
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
- [STATUS](STATUS.md) is the sole source for current execution status. Other
  files may define contracts/procedures or retain clearly labeled history, but
  must not act as a parallel current-status register.

## Archive and reports

Files under [archive](archive/) are historical or superseded references. They
are not authoritative and are not part of the default reading path.

The three DOCX files in `docs/` are reports for the instructor. This handbook
does not replace, rename, or supersede those reports.
