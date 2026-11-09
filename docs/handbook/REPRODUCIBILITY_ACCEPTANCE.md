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
| 1 Quick Start | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 1; `src/dataset_b/reproduce/`; focused unit tests; recorded 2026-11-03 author-machine observations and 2026-11-09 partial clean-worktree PASS in STATUS | public download and asset verification passed in a detached worktree, but environment, evaluation, comparison, research, and reports were not run there | run the pinned complete clean-clone route on a qualified host |
| 2 Repository structure | `PASS` | `README.md`; `config/README.md`; `notebooks/dataset_b/README.md`; `src/dataset_b/README.md`; handbook router | none for documented Dataset B navigation | none |
| 3 Hardware and software | `PASS_WITH_LIMITATION` | accepted RQ1 runtime/manifests; RQ4 CPU runtime manifest; STATUS | RTX 5050 Laptop GPU, nominal 8 GB device memory where evidenced, and CUDA build 13.0 are recorded; CPU model, RAM, driver, cuDNN, and some exact OS/Python fields remain unrecorded | none for current acceptance; do not rerun CUDA smoke solely to fill fields |
| 4 Environment installation | `PASS_WITH_LIMITATION` | `environments/reproducibility/`; `requirements.txt`; `pyproject.toml`; recorded 2026-11-03 author-machine installation | the five profiles were installed on the author machine, but no clean-host resolver/install sequence was executed; CPU and GPU profiles differ and no lower compatibility floor is inferred | run the pinned install route on a clean qualified host |
| 5 Data | `PASS_WITH_LIMITATION` | data/split contracts; root locks; six frozen run manifests; 2026-11-09 public-link and partial clean-worktree checks in STATUS | raw HMDA is optional provenance and upstream terms are not declared in repository evidence | none for the locked public split and bundle identities; retain the recorded links and hashes |
| 6 Config and hyperparameters | `PASS` | `config/dataset_b/catalog.yaml` and its eleven targets; accepted manifests | config status may intentionally lag accepted STATUS for frozen lineage | none; interpret frozen config with STATUS |
| 7 Training | `PASS_WITH_LIMITATION` | `REPRODUCE.md` Section 7; six accepted frozen RQ1 run IDs; three completed RQ4b development runs; STATUS | the inventory records only completed accepted runs and actual seed coverage; no fresh training occurred in this task and several future routes are internal/operator-only | none for current documentation acceptance; fresh training requires a separate task |
| 8 Evaluation and inference | `DOCUMENTED_NOT_RERUN` | `REPRODUCE.md` Sections 1 and 8; split/evaluation contract; accepted Test manifests; STATUS | historical checkpoint evaluation was observed before the final source edits, but the pinned current source was not rerun end to end; RQ2 was not run | rerun only the pinned frozen-checkpoint route on a qualified host; no training or fitting |
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

On 2026-11-03 the owner decided that the project bundles would be distributed through
Google Drive links, which were not yet available at that date. On that basis the owner accepted
gate G1 and authorized work on the following gates. This is an owner decision, not
download evidence: at that date the links were not recorded, anonymous download through them was
not verified, and anonymous repository access returned HTTP 401. The
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

On 2026-11-09 the owner supplied byte-preserving links for the three locked split CSVs
and links for eleven core ZIPs. The manifest treats the CSVs as independent direct
downloads and retains raw HMDA only as optional historical provenance. Implementation
tests passed 119/119 and all eleven local ZIP identities and member inventories passed,
totaling 678,068,350 bytes. Anonymous publication verification passed for the
repository endpoint, all three exact split identities, and the expected remote sizes
of all eleven ZIPs. A detached temporary worktree then downloaded the CSVs and ZIPs;
`download` and `verify-assets` both passed with 653/654 assets verified, no mismatch or
required asset missing, and one optional provenance asset absent. This closes the
public-download and partial clean-worktree gates, not CUDA Phase B or the full
clean-clone route.

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
| TG-1 | 5, 8 | Data description, preprocessing steps, and metric definitions | written in `REPRODUCE.md` Sections 5.3, 5.4, and 8.1; metric formulas are read from the LR route only; `download` obtained the three locked split CSVs directly in the partial clean-worktree PASS, and later preprocessing runs inside the model routes | none for the data-download gate; model-route execution remains separate |
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
| Public data and checkpoint links with checksums | met for the 3 locked CSVs and 11 core bundles | Sections 5.1, 5.2, 5.5; manifest; retained partial clean-worktree receipts |
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

The current-host technical issue recorded on 2026-11-05 is resolved and `COMPLETE`:
the asset gate passed 654/654; this machine has no CUDA and does not contain the five
locked virtual environments; the owner records that those environments are on another
machine; and the owner stopped the current Phase B and Phase C attempt. No checkpoint
or metric comparison failed in this attempt.

The current machine has no CUDA and does not contain the locked virtual environments;
the owner records that those environments are on another machine. Public publication
and partial clean-worktree download/asset verification passed on 2026-11-09. This
acceptance does not establish a successful clean-clone run, current-source end-to-end
execution, notebook regeneration, or universal runtime reproducibility. Completion of
the public-download gate does not mark the entire reproduction runtime complete.

This conclusion is a technical issue-resolution record. It is not a training result,
a new model evaluation result, a current-host GPU compatibility result, or a universal
numerical reproducibility claim.
