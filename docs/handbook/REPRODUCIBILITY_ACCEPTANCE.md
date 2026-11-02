# Dataset B Reproducibility Documentation Acceptance

## Authority and scope

This register accepts the completeness and evidence discipline of Dataset B
reproducibility documentation. It does not replace [STATUS](STATUS.md), which
remains the sole register of experimental execution state.

- Reviewed submission commit: PR #12 merge commit
  `369fe494506fce888d6c917f504d086a1dfcb097`.
- Reviewed PR head: `ed1afbf783db502ade5dfb959738df51cdb6a5e7`.
- Documentation version/date: `2.0`, `2026-11-02`.
- Scope: the rubric sections 0 through 12 in
  [REPRODUCE](../../REPRODUCE.md), Dataset B only.
- Evidence rule: repository source, Git history, accepted historical manifests,
  receipts, frozen artifacts, reports, saved notebook structure, and handbook.
- Runtime boundary: no environment installation, training, checkpoint load,
  model inference, evaluation backend, notebook execution, or GPU execution was
  performed for this acceptance.

## Status vocabulary

- `PASS`: required documentation is present and statically verified.
- `PASS_WITH_LIMITATION`: documentation is present but the underlying runtime
  or distribution evidence has an explicit limitation.
- `DOCUMENTED_NOT_RERUN`: a source-backed procedure is documented but was not
  executed in this task.
- `NOT_VERIFIED`: the repository lacks enough accepted evidence for the claim.
- `NOT_APPLICABLE`: the rubric item does not apply to the documented scope.

## Rubric 0 through 12 matrix

| Item | Status | Evidence path | Limitation | Action required to raise status |
|---:|---|---|---|---|
| 0 General information | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 0; Git commits `369fe494...` and `ed1afbf...` | formal bilingual title, semester, supervisor, roster, IDs, and contact data are not declared | add only authoritative project metadata |
| 1 Quick Start | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 1; `src/dataset_b/cli.py`; `src/dataset_b/core/navigation.py` | no accepted clean-host install; retained route needs local non-Git artifacts | complete and retain a separately authorized clean-host walkthrough and artifact handoff |
| 2 Repository structure | `PASS` | `README.md`; `config/README.md`; `notebooks/dataset_b/README.md`; `src/dataset_b/README.md`; handbook router | none for documented Dataset B navigation | none |
| 3 Hardware and software | `PASS_WITH_LIMITATION` | accepted RQ1 runtime/manifests; RQ4 CPU runtime manifest; STATUS | CPU model, RAM, driver, cuDNN, and some exact OS/Python fields are not recorded | capture those fields in a future authorized runtime manifest |
| 4 Environment installation | `DOCUMENTED_NOT_RERUN` | `requirements.txt`; `pyproject.toml`; model runbooks | no clean resolver/install evidence; CPU and GPU profiles differ | run an authorized clean-host install and preserve logs/fingerprint |
| 5 Data | `PASS_WITH_LIMITATION` | data/split contracts; root locks; six frozen run manifests | raw data and retained artifacts are not Git-distributed; upstream terms are not declared in repository evidence | publish an authorized artifact/data handoff with checksums and terms |
| 6 Config and hyperparameters | `PASS` | `config/dataset_b/catalog.yaml` and its eleven targets; accepted manifests | config status may intentionally lag accepted STATUS for frozen lineage | none; interpret frozen config with STATUS |
| 7 Training | `DOCUMENTED_NOT_RERUN` | baseline, neural, and TabMmini runbooks; six accepted frozen run IDs | no training occurred in this task; several routes are internal/operator-only | separately authorize and retain a fresh training reproduction |
| 8 Evaluation and inference | `DOCUMENTED_NOT_RERUN` | split/evaluation contract; accepted Test manifests; RQ2/RQ3/RQ4 evidence | no checkpoint was loaded and no model-facing evaluation/inference occurred in this task | separately authorize retained-checkpoint evaluation on a qualified host |
| 9 Seeds and determinism | `PASS_WITH_LIMITATION` | split/model/RQ configs; numerical comparison policies; accepted multi-seed RQ4b evidence | cross-host/GPU determinism is not established and tolerances are model-specific | repeat under a declared host matrix without changing locked policies |
| 10 Result/report mapping | `PASS_WITH_LIMITATION` | final notebook router; report index; RQ1-RQ4 accepted run roots | notebooks were not executed; mapping relies on retained accepted evidence | run an authorized report rebuild and compare durable outputs |
| 11 Limitations and troubleshooting | `PASS` | `REPRODUCE.md` Section 11; runbooks; STATUS limitations | none for the documentation claim | none |
| 12 License citation and AI | `PASS_WITH_LIMITATION` | Git history; STATUS; RQ4 plan/runbook; repository file inventory | project license, citation metadata, and formal upstream citations are not declared | owners must add authoritative license/citation metadata |

## Verification gates

Documentation acceptance requires all of the following:

1. Sections 0 through 12 are present once and in order.
2. Internal Markdown links resolve to files in the repository.
3. The submission merge and reviewed head identities are exact.
4. The five public Dataset B CLI commands are documented and remain read-only.
5. Acceptance matrix statuses use only the vocabulary above.
6. The documents do not claim clean-install, training, model/GPU execution, or
   train-from-scratch success in this task.
7. The handbook router points to this register while STATUS remains the sole
   experimental execution register.
8. No public navigation command exposes train, evaluate, predict, inference,
   or serve.

## Acceptance conclusion

The documentation target is
`REPRODUCIBILITY_DOCUMENTATION_ACCEPTANCE_COMPLETE` when the static tests,
five read-only CLI checks on retained local evidence, diff/scope inspection,
and delivery gates all pass.

This conclusion is documentation acceptance only. It is not a clean-host
runtime result, model evaluation result, training result, GPU compatibility
result, or universal numerical reproducibility claim.
