# Dataset B Reproducibility Documentation Acceptance

## Authority and scope

This register accepts the completeness and evidence discipline of Dataset B
reproducibility documentation. It does not replace [STATUS](STATUS.md), which
remains the sole register of experimental execution state.

- Reviewed submission commit: PR #12 merge commit
  `369fe494506fce888d6c917f504d086a1dfcb097`.
- Reviewed PR head: `ed1afbf783db502ade5dfb959738df51cdb6a5e7`.
- Documentation version/date: `2.3`, `2026-11-03`.
- Source version: commit `4bdfa0a5aebe138982b75a055891c1321a4d5496`
  on `main`.
- Scope: the rubric sections 0 through 12 in
  [REPRODUCE](../../REPRODUCE.md) version 2.3, Dataset B only.
- Evidence rule: repository source, Git history, accepted historical manifests,
  receipts, frozen artifacts, reports, saved notebook structure, and handbook.
- Runtime boundary: this documentation review did not itself perform a new clean
  installation, training run, checkpoint load, model inference, notebook execution,
  or end-to-end clean-clone run. Historical author-machine observations of the
  reproduction route are recorded separately in REPRODUCE Section 1 and
  [STATUS](STATUS.md).

## Recorded acceptance decisions

The following owner decisions define the current documentation-acceptance gate:

1. Quick Start implements frozen-checkpoint evaluation and comparison against the
   expected headline metrics. Parts of the route were historically observed on the
   author machine, but the complete clean-clone sequence was not executed and the
   current source was not rerun end to end after its final edits.
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
| 0 General information | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 0; source commit `4bdfa0a5...`; submission commits `369fe494...` and `ed1afbf...` | formal bilingual title, semester, supervisor, roster, IDs, and contact data are not declared | add only authoritative project metadata |
| 1 Quick Start | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 1; `src/dataset_b/reproduce/`; source commit `4bdfa0a5...`; focused unit tests; 2026-11-08/09 qualified-host receipts in STATUS | the full qualified-host sequence through `reports` was executed, but it was not a clean clone and bundle download was not verified | run the pinned clean-clone route after the final tag and bundle links exist |
| 2 Repository structure | `PASS` | `README.md`; `config/README.md`; `notebooks/dataset_b/README.md`; `src/dataset_b/README.md`; handbook router | none for documented Dataset B navigation | none |
| 3 Hardware and software | `PASS_WITH_LIMITATION` | accepted RQ1 runtime/manifests; RQ4 CPU runtime manifest; `environment_20261108T125329_497165Z.json`; STATUS | the 2026-11-08 host verified RTX 5050 Laptop GPU, 8,546,484,224 bytes, driver 610.74, Torch 2.14.0+cu130, CUDA build 13.0, cuDNN 92400, and finite CUDA operations in all three neural profiles; CPU model and RAM remain unrecorded | record CPU/RAM only from authoritative evidence; do not infer a lower compatibility floor |
| 4 Environment installation | `PASS_WITH_LIMITATION` | five locks under `environments/reproducibility/`; `environment_20261108T125329_497165Z.json` | all five profiles installed and passed on the qualified host; this was an in-place handoff-host installation, not a clean-clone installation | rerun only as part of the final clean-clone gate after public links and final revision exist |
| 5 Data | `PASS_WITH_LIMITATION` | data/split contracts; root locks; six frozen run manifests; latest 654/654 receipt; final Phase C bundle verification record; release commit `1d38448...` | 654 retained assets and all 11 core ZIPs passed identity/member checks and were pushed through Git LFS; the default route now uses three locked direct-download CSVs and keeps raw HMDA as optional provenance, but the URL fields remain null, so anonymous distribution is unverified | supply the actual HTTPS URLs and verify anonymous download plus clean-clone installation |
| 6 Config and hyperparameters | `PASS` | `config/dataset_b/catalog.yaml` and its eleven targets; accepted manifests | config status may intentionally lag accepted STATUS for frozen lineage | none; interpret frozen config with STATUS |
| 7 Training | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 7; six accepted frozen RQ1 run IDs; three completed RQ4b development runs; STATUS | the inventory records only completed accepted runs and actual seed coverage; no fresh training occurred in this task and several future routes are internal/operator-only | none for current documentation acceptance; fresh training requires a separate task |
| 8 Evaluation and inference | `PASS_WITH_LIMITATION` | `evaluate_20261108T125910_329422Z.json`; `compare_20261108T125939_503172Z.json`; `research_20261108T174830_904582Z.json`; STATUS | six frozen-checkpoint Test evaluations, 402 RQ1 rows, and every RQ2-RQ4 route passed; this was an in-place qualified-host run rather than a clean clone | retain the receipts and repeat only in the final clean-clone gate |
| 9 Seeds and determinism | `PASS_WITH_LIMITATION` | split/model/RQ configs; numerical comparison policies; accepted multi-seed RQ4b evidence; `research_20261108T174830_904582Z.json`; STATUS | locking TabMmini frozen CPU inference to one intra-op thread restored all 28 RQ2 comparisons without reselection or policy change; the accepted historical package still lacks enough provenance to prove its original thread configuration | retain both the initial failed receipt and passing rerun as diagnostic evidence |
| 10 Result/report mapping | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 10; `reports_20261108T174846_986881Z.json`; report mapping; retained reports and figure notes | five mapped objects passed; Figures 2 and 3 plus assembled RQ text reports remain `NOT_REGENERATED` because no source-backed builder exists and notebooks were not executed | none for current Phase B acceptance; notebook regeneration requires a separately authorized expanded scope |
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

On 2026-11-05 the owner separately authorized preservation of selected RQ4 historical
review evidence. The resulting `rq4_audit_evidence_manifest.json` is independent of
the core asset manifest: it describes an optional audit archive, not a required
reproduction bundle. `scratch/` remains ignored local state, the original files were
not deleted or moved, and the archive URL is not yet recorded. This preservation work
does not raise the current runtime status or establish public download verification.

The same review reconciled RQ3 `runtime.csv` as volatile historical timing metadata,
not a scientific reproduction input. The canonical file remains in its accepted run,
while the core manifest and rebuilt RQ3 bundle exclude it. The manifest now records an
included-inventory digest, source revision, and the excluded volatile path. Local
asset verification passed 654/654; public bundle publication and anonymous download
remain separate delivery gates.

Later on 2026-11-03 the owner prohibited training or fitting any model and then set
the acceptance criterion for the remaining work: the command-line route must work
technically, with no process run again. As a result, the RQ4a route validates the
stored packages without fitting its post-processor, the RQ2 route was not run, and
the clean-clone gate was not run. These are recorded limits, not reproduction results.

## Reproducibility guide template coverage plan

On 2026-11-03 the owner added to the reproducibility plan the course reproducibility
guide template requirements that a comparison on that date found outside the existing
delivery gates. The list reflects that comparison and is not a proof of exhaustive
coverage. This table records how version 2.3 addresses those requirements; it is not
execution status and does not replace the matrix or STATUS. The template expects Quick
Start to reproduce the main result, while the current evidence still stops short of a
complete clean-clone execution.

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
| Commit or tag and links in Section 0 | commit met; tag and links not met | source commit `4bdfa0a5...` is pinned; the final tag and bundle links come from the owner at delivery |
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
run manifests. Version 2.3 documentation coverage is accepted with the limitations
recorded here; the delivery gates remain conditions for full clean-clone runtime
reproducibility. The stale commands in `docs/handbook/runbooks/neural_training.md`
(lines 625 to 628) remain; that file is outside the current write scope.

## Historical acceptance

Version 2.1 received documentation-only acceptance on 2026-11-02. It was
superseded for current documentation coverage by the version 2.3 review after the
reproduction CLI, environment profiles, asset manifest, and expanded Quick Start were
added. This historical decision is not current execution status.

## Acceptance conclusion

The CUDA handoff Phase B run on 2026-11-08 through 2026-11-09 is complete within its
qualified-host scope. Asset integrity passed
654/654 at commit `5a5667b13546e6b088756cd8e9da8e963a0ccfe0`. All five locked
environment profiles passed; the three neural profiles performed finite CUDA tensor
operations. Six frozen-checkpoint evaluations and all 402 RQ1 comparisons passed.
RQ3, RQ4a, and RQ4b also passed without training or fitting.

An initial RQ2 run failed three TabMmini rows. Controlled frozen inference identified
the unpinned Torch CPU thread count as the current-host cause; a scoped one-thread lock
then produced a 28/28 passing official rerun. The tolerance and accepted artifacts
remain unchanged. `reports` was run and returned the documented `INCOMPLETE` status:
five objects passed and three notebook/authored-report objects were not regenerated.
Phase C then rebuilt all 11 core ZIPs. All bundle identities and member inventories
matched the prebuild manifest; RQ3 volatile runtime metadata and the optional RQ4 audit
archive remained outside core. All 11 core LFS objects were pushed in release commit
`1d38448...`; Phase C build/stage/commit/push is complete. Clean-clone and anonymous
public-download verification remain unexecuted; bundle links and the three direct
split-CSV URLs were not supplied.

This acceptance does not establish a successful clean-clone end-to-end run,
notebook regeneration, or universal runtime reproducibility.

This conclusion is a gate-status record. It is not a training result, a new model
evaluation result, a locked-profile GPU compatibility result, or a universal numerical
reproducibility claim.
