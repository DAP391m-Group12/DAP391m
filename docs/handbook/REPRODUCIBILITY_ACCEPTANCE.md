# Dataset B Reproducibility Documentation Acceptance

## Authority and scope

This register accepts the completeness and evidence discipline of Dataset B
reproducibility documentation. It does not replace [STATUS](STATUS.md), which
remains the sole register of experimental execution state.

- Reviewed submission commit: PR #12 merge commit
  `369fe494506fce888d6c917f504d086a1dfcb097`.
- Reviewed PR head: `ed1afbf783db502ade5dfb959738df51cdb6a5e7`.
- Documentation version/date: `2.1`, `2026-11-02`.
- Scope: the rubric sections 0 through 12 in
  [REPRODUCE](../../REPRODUCE.md), Dataset B only.
- Evidence rule: repository source, Git history, accepted historical manifests,
  receipts, frozen artifacts, reports, saved notebook structure, and handbook.
- Runtime boundary: no environment installation, training, checkpoint load,
  model inference, evaluation backend, notebook execution, or GPU execution was
  performed for this acceptance.

## Recorded acceptance decisions

The following owner decisions define the current documentation-acceptance gate:

1. Quick Start validates repository navigation, saved reporting structure, and
   retained locks. It is not required to rerun a model or recalculate a headline
   metric.
2. Accepted GPU documentation records the NVIDIA GeForce RTX 5050 Laptop GPU,
   nominal 8 GB device memory where the run evidence contains the exact byte
   count, and CUDA build 13.0. No new CUDA smoke is required.
3. Each accepted run keeps its exact package versions. A package may be labeled
   with a lowest verified working version only when successful retained evidence
   exists at that version; this is not an inferred general compatibility floor.
4. Training documentation lists only completed accepted runs and their actual
   seed coverage. Fresh training is not part of this acceptance.
5. RQ-level mapping plus object-level provenance retained in the reports and
   figure verification notes satisfies the current result/report mapping gate.
   A fresh report rebuild is outside scope.
6. License and citation completeness is excluded from the acceptance score.
   `NOT_DECLARED` inventory remains for transparency; the AI-use declaration
   remains in scope.

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
| 1 Quick Start | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 1; `src/dataset_b/cli.py`; `src/dataset_b/core/navigation.py` | accepted scope verifies repository/report navigation and retained locks; it does not recalculate a headline metric, and the retained route needs local non-Git artifacts | none for current documentation acceptance; a clean-host artifact handoff is required only if scope expands |
| 2 Repository structure | `PASS` | `README.md`; `config/README.md`; `notebooks/dataset_b/README.md`; `src/dataset_b/README.md`; handbook router | none for documented Dataset B navigation | none |
| 3 Hardware and software | `PASS_WITH_LIMITATION` | accepted RQ1 runtime/manifests; RQ4 CPU runtime manifest; STATUS | RTX 5050 Laptop GPU, nominal 8 GB device memory where evidenced, and CUDA build 13.0 are recorded; CPU model, RAM, driver, cuDNN, and some exact OS/Python fields remain unrecorded | none for current acceptance; do not rerun CUDA smoke solely to fill fields |
| 4 Environment installation | `DOCUMENTED_NOT_RERUN` | `requirements.txt`; `pyproject.toml`; accepted run package records; model runbooks | no clean resolver/install evidence; CPU and GPU profiles differ; exact per-run versions are preserved and no lower compatibility floor is inferred | none for current acceptance; a clean-host install is a separate expanded scope |
| 5 Data | `PASS_WITH_LIMITATION` | data/split contracts; root locks; six frozen run manifests | raw data and retained artifacts are not Git-distributed; upstream terms are not declared in repository evidence | publish an authorized artifact/data handoff with checksums and terms |
| 6 Config and hyperparameters | `PASS` | `config/dataset_b/catalog.yaml` and its eleven targets; accepted manifests | config status may intentionally lag accepted STATUS for frozen lineage | none; interpret frozen config with STATUS |
| 7 Training | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 7; six accepted frozen RQ1 run IDs; three completed RQ4b development runs; STATUS | the inventory records only completed accepted runs and actual seed coverage; no fresh training occurred in this task and several future routes are internal/operator-only | none for current documentation acceptance; fresh training requires a separate task |
| 8 Evaluation and inference | `DOCUMENTED_NOT_RERUN` | split/evaluation contract; accepted Test manifests; RQ2/RQ3/RQ4 evidence | no checkpoint was loaded and no model-facing evaluation/inference occurred in this task | separately authorize retained-checkpoint evaluation on a qualified host |
| 9 Seeds and determinism | `PASS_WITH_LIMITATION` | split/model/RQ configs; numerical comparison policies; accepted multi-seed RQ4b evidence | cross-host/GPU determinism is not established and tolerances are model-specific | repeat under a declared host matrix without changing locked policies |
| 10 Result/report mapping | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 10; final notebook router; RQ1/RQ2/RQ3-RQ4 retained reports; figure verification notes; accepted run roots | RQ-level mapping and retained object-level provenance are accepted; notebooks and report generation were not rerun | none for current acceptance; per-object regeneration commands are needed only for an expanded fresh-report scope |
| 11 Limitations and troubleshooting | `PASS` | `REPRODUCE.md` Section 11; runbooks; STATUS limitations | none for the documentation claim | none |
| 12 License citation and AI | `PASS` | `REPRODUCE.md` Section 12; Git history; STATUS; RQ4 plan/runbook; repository file inventory | license/citation completeness is excluded from the acceptance score; `NOT_DECLARED` inventory is retained transparently; AI-use evidence remains recorded | none for current documentation acceptance |

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

## Asset distribution decision

On 2026-11-03 the owner decided that the project bundles will be distributed through
Google Drive links, which the owner will add later. On that basis the owner accepted
gate G1 and authorized work on the following gates. This is an owner decision, not
download evidence: the links are not yet recorded, anonymous download through them is
not verified, and anonymous repository access returned HTTP 401 when last checked. The
clean-clone gate needs the links recorded and every download hash-verified.

Later on 2026-11-03 the owner prohibited training or fitting any model and then set
the acceptance criterion for the remaining work: the command-line route must work
technically, with no process run again. As a result, the RQ4a route validates the
stored packages without fitting its post-processor, the RQ2 route was not run, and
the clean-clone gate was not run. These are recorded limits, not reproduction results.

## Reproducibility guide template coverage plan

On 2026-11-03 the owner added to the reproducibility plan the course reproducibility
guide template requirements that a comparison on that date found outside the existing
delivery gates. The list reflects that comparison and is not a proof of exhaustive
coverage. This table is a plan, not execution status, and it does not change the
matrix or the recorded decisions above, which define documentation acceptance only.
Those decisions are less strict than the template in places: the template expects
Quick Start to reproduce the main result. The matrix was reviewed against
`REPRODUCE.md` version `2.1`, so it does not cover the sections added in version `2.2`.

| ID | Template section | Requirement | State | Condition to proceed |
|---|---|---|---|---|
| TG-1 | 5, 8 | Data description, preprocessing steps, and metric definitions | written in `REPRODUCE.md` Sections 5.3, 5.4, and 8.1; metric formulas are read from the LR route only; `download` rebuilds the split CSVs, and later preprocessing runs inside the model routes | none |
| TG-2 | 4 | Declare environment variables | written in `REPRODUCE.md` Section 4; no `.env.example` is needed | none |
| TG-3 | 9 | Multi-seed results for RQ1 | declared as a limitation in `REPRODUCE.md` Sections 9 and 11 | new training needs a separate owner authorization |
| TG-4 | 7 | Training command per model, outside the read-only navigation CLI | written in `REPRODUCE.md` Section 7.2 as `DOCUMENTED_NOT_RERUN`; LR and TabMmini have no command-line training route; a reduced run is not required because accepted fits are under 4 GPU hours | execution needs a separate owner authorization |
| TG-5 | 8 | Inference command for new data, outside the read-only navigation CLI | `predict` implemented with synthetic unit tests; not run on a frozen checkpoint | none for basic acceptance |
| TG-6 | 3 | Minimum configuration, disk, and evaluation time | used configuration, disk, and time written in `REPRODUCE.md` Section 3.2; no lower floor was tested; CPU and RAM are not recorded | none for basic acceptance |
| TG-7 | 2 | Notebook `Restart & Run All` evidence | not verified | notebook execution needs a separate owner authorization |
| TG-8 | 0, 12 | Formal titles, roster, supervisor, code license, and citation | owner input required | the owner supplies the values |

### Template checklist review, 2026-11-03

The owner set the scope of this review on 2026-11-03: basic acceptance of `REPRODUCE.md`
version `2.3` against the template, without counting process execution. `MET` means the
guide contains the required content; it is not a runtime result.

| Checklist item | State | Evidence or gap |
|---|---|---|
| Commit or tag and links in Section 0 | not met | branch is named; the tag and bundle links come from the owner at delivery |
| Quick Start from clone to expected numbers | met in content; clean run not executed | `REPRODUCE.md` Section 1.0 |
| Exact version locks installable in a new environment | met; installed on the author machine only | `environments/reproducibility/` |
| Public data and checkpoint links with checksums | checksums met; links not met | Sections 5.1, 5.2, 5.5; manifest |
| No absolute personal paths in code | met for `src`, `scripts`, and `tests`; the Colab profile config holds three `/content` paths | a search of tracked files on 2026-11-03 |
| Fixed Test set not used for selection | met | `split_v1.json`; split and evaluation contract |
| Seeds recorded; multi-seed mean and standard deviation | seeds met; multi-seed not met for RQ1, declared as a limitation | Section 9 |
| Every report table and figure has a mapping row | met with gaps | Section 10.1 maps report sections, tables, and figures; many have no regeneration command, Figures 4 to 6 are absent from the repository, Figure 1 has no regeneration route, and tables were not checked cell by cell |
| Baseline also reproducible | met in content; observed 2026-11-03 | Sections 1.0 and 8 |
| No committed secrets; `.env.example` | no secrets found; `.env.example` not needed | Section 4 |
| Notebooks run with `Restart & Run All` | not verified | saved structure is checked; execution was not authorized |
| Non-reproducible parts and AI tools declared | met | Sections 11 and 12 |

Open items that only the owner can supply: formal titles, semester, supervisor, roster,
student IDs, contact, code license, HMDA data terms, the final tag, and the 11 bundle
links.

Other known gaps: CPU and RAM are not recorded; peak training VRAM is not recorded for
the final fits; LR and TabMmini have no command-line training route; the Quick Start has
12 commands against the template's guide of 5 to 10; and the code was edited after the
last observed `evaluate`, `compare`, and `reports` runs.

On 2026-11-03 the owner authorized correcting the two stale locks in
`src/dataset_b/core/navigation.py`; they now hold the LF values pinned by the accepted
run manifests. `REPRODUCIBILITY_DOCUMENTATION_ACCEPTANCE_COMPLETE` still depends on the
delivery gates. The stale commands in `docs/handbook/runbooks/neural_training.md` (lines
625 to 628) remain; that file is outside the current write scope.

## Acceptance conclusion

The documentation target is
`REPRODUCIBILITY_DOCUMENTATION_ACCEPTANCE_COMPLETE` when the static tests,
five read-only CLI checks on retained local evidence, diff/scope inspection,
and delivery gates all pass.

This conclusion is documentation acceptance only. It is not a clean-host
runtime result, model evaluation result, training result, GPU compatibility
result, or universal numerical reproducibility claim.
