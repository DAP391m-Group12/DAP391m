# Documentation migration record — September 2026

> Archive record. This file documents the completed documentation migration;
> it is not an agent's default reading path or an execution/status source.
> For active work, follow the task-specific contracts and runbooks linked by
> the current documentation entry point.

## Purpose

This record preserves the migration map and G1–G4 history from the former
`docs/README.md`. It explains where the previous Markdown responsibilities
were placed in the handbook and records the boundaries and checks applied
during the rewrite. It does not replace the current ownership contracts.

## G1 — content migration map

The controlled rewrite retained or handed off the content as follows:

| Former source and heading group | Destination / treatment |
| --- | --- |
| `Dataset_B_Data_Selection.md` — purpose and interpretation boundary | Data contract: cohort, target, and rule that transaction flags do not filter the cohort. |
| Data Selection — 99-column reconciliation | Data contract: `35 + 9 + 1 + 54 = 99`, role partition, and `record_id` boundary. |
| Data Selection — retained columns and target | Data contract: target source, nine audit columns, 35-candidate inventory, and raw-versus-derived alternatives; feature decisions are linked rather than duplicated. |
| Data Selection — excluded columns | Data contract: complete Gate 1–2 exclusion list and reasons. |
| Data Selection — required checks | Data contract: schema, lineage, timing, applicability, and leakage gates; completed feature evidence is linked to Feature Selection. |
| Data Selection — former experiment design and decision rules | Feature-governance protocol, sections 10.1–12; Data Selection keeps a handoff link. |
| Data Selection — former current status and references | Contract-level status and source register in the current Data Selection document. |
| `Feature_selection.md` — purpose, target, and grouping | Feature-governance introduction, condensed to a Data Contract reference. |
| Feature Selection — priority screening | Feature-governance evidence: 12 `keep`, one `exclude_from_main`, and two representation-comparison statuses. |
| Feature Selection — candidates requiring checks | Feature-governance evidence: candidate-block evidence, timing gates, and unresolved decisions. |
| Feature Selection — representation alternatives | Feature-governance decision gate for mutually exclusive raw-versus-derived comparisons. |
| Feature Selection — metadata | Data-contract cross-reference; the canonical nine-column inventory remains in Data Selection. |
| Feature Selection — audit, redundancy, sensitivity, and decision rules | Feature-governance protocol and methods. |
| Feature Selection — notebook outputs | Execution contract: required artifacts and separation of screening from full training. |
| Feature Selection — decision state and related documents | Feature-governance decision state and source register. |
| `Dataset_B_v1_preprocessing.md` — handoff status, data boundary, and 12-feature contract | Classical preprocessing contract: imported feature/split versions, exact raw input order, target/audit separation, and Train-only fit boundary. Current model execution status is owned by `handbook/STATUS.md`, not by the preprocessing source. |
| Classical preprocessing — parser, numeric two-channel representation, fixed OHE, and model profiles | Classical preprocessing contract: field-specific token semantics, four value/status pairs, 66-column schema, LR scaling, and RF/XGBoost no-scaling profiles. |
| Classical preprocessing — exported CV artifact | Classical preprocessing contract: fold-local matrix schema, row/label/identifier alignment, artifact limitations, and preservation rules. Observed model outcomes are excluded from the contract. |
| Classical preprocessing — notebook setup and CV/full-Train examples | Baseline training runbook: concise use of the supported loaders, pipeline factory, exported matrices, full-Train fit, Validation prediction, and Test handoff. Long notebook examples are not duplicated. |
| Classical preprocessing — threshold, logging, checklist, and limitations | Split/evaluation contract plus baseline runbook: threshold/Test boundaries and operational checks. Current execution outcomes are owned only by `handbook/STATUS.md`. |
| `Dataset_B_Engine_preprocesing.md` — purpose and shared data boundary | Classical and neural preprocessing contracts: common 12-input boundary, parser semantics, split-fit rules, and separation between project semantic preprocessing and model execution. |
| Engine preprocessing — Pipeline A for LR/RF/XGBoost | Classical preprocessing contract: 66-column external-OHE route and model-specific scaling behavior. |
| Engine preprocessing — Pipeline B semantic layer and categorical policy | Neural preprocessing contract: 16-column semantic frame, fold-fit median then `log1p`, fixed category meanings, category-support gate, and dtype/container rules. |
| Engine preprocessing — PlainMLP, RealMLP, and TabMmini adapters | Neural preprocessing contract: PlainMLP external-OHE route and separate native RealMLP/TabMmini routes. Model-specific runtime procedure is handed to the neural runbook. |
| Engine preprocessing — regularization, versions, artifact contracts, matrix, and acceptance gates | Neural preprocessing contract for preprocessing artifacts and acceptance criteria; model controls remain model configuration. Execution outcomes are not migrated into the contract. |
| Engine preprocessing — dated implementation/smoke evidence | Current execution claims are handed to `handbook/STATUS.md`; only explicitly dated historical context remains in the former source. The handbook contracts do not use that snapshot as proof of current completion. |
| `Dataset_B_Pipeline.md` — PlainMLP task chain | Neural training runbook for stage order and gates; current PMLP evidence and open stages are handed to `handbook/STATUS.md`. |
| Pipeline — dated RealMLP stage history | `handbook/STATUS.md` for current state. The former table remains historical source material and is not duplicated as a second live register. |
| Pipeline — Dataset B flow and execution modes | Split/evaluation contract for partition and RQ boundaries; baseline and neural runbooks for operational procedures. The architecture overview is condensed rather than copied. |
| Pipeline — preprocessing routes and neural wrapper boundary | Classical and neural preprocessing contracts for model-family inputs; the neural runbook owns runtime orchestration. |
| Pipeline — RQ1 decision register, split/CV/refit/Validation/Test rules | Split/evaluation contract: immutable split, Train-only CV, fresh full-Train refit, threshold selection, Test evaluation, bootstrap, and comparison boundaries. |
| Pipeline — policy gates, artifact handoff, code mapping, and task routing | Contracts and runbooks by responsibility. Current model-stage outcomes and outstanding reviews are handed to `handbook/STATUS.md`. |
| `Neural_Training_Guideline.md` — authority, machine responsibilities, data boundary, and model order | Neural training runbook: local GPU workflow, authorization envelope, Train/Validation/Test boundary, required sequence, and RealMLP-before-TabMmini rule. |
| Neural guideline — shared preprocessing and native adapters | Neural preprocessing contract: shared semantic layer, native input boundary, dtype/category requirements, and separation from PlainMLP OHE. |
| Neural guideline — pilot split, provisional configuration, and GPU gate | Neural training runbook: immutable Train-only pilot, runtime compatibility checks, pilot procedure, and research-decision boundary. Exact model parameters remain in versioned config. |
| Neural guideline — observability, history, and epoch-policy decision | Neural training runbook: durable local history, required diagnostics, and fixed-epoch versus Train-only inner-stopping procedure. |
| Neural guideline — atomic publication, resume, downstream stages, and handoff | Neural training runbook: temporary/final directory lifecycle, recovery, CV/full-Train/Validation/Test handoff, and operator checklist. |
| Neural guideline — dated runtime/CV evidence and locked decision register | Current outcomes are handed to `handbook/STATUS.md`. Contract-level decision states may remain in the runbook only when they describe policy rather than execution completion. |
| `README.md` — purpose, authority, ownership, and reading path | Handbook README: task-based reading path, one owner per responsibility, STATUS authority, archive warning, and preservation of instructor reports. |
| README — G1–G4 migration history and reference checks | This archive record. Historical checks are retained as historical claims and do not become fresh verification. |
| `Colab_env.md` — superseded status and redirect | `archive/colab_environment_superseded.md`: archived redirect with an explicit non-runtime banner and a pointer to the active neural runbook. |

The table covers all eight former Markdown sources. Lower-level headings remain
with their listed parent group unless the map records an explicit handoff.
Content migration does not itself authorize handbook cutover or deletion of
the former sources.

### Feature-notebook attribution correction

The former `Feature_selection.md` and the former README attributed the priority
screening narrative to `notebooks/eda_datasetB/05_feature_selction.ipynb`.
The later content review found that the saved evidence is split across two
notebooks:

- `04_feature_selection.ipynb` contains the executed priority screening for
  groups 4.1–4.6;
- `05_feature_selction.ipynb` contains the executed checkpoint for review
  groups 5.1–5.3.

The handbook feature-governance contract uses this corrected attribution. The
correction identifies provenance only; it does not claim either notebook was
rerun during documentation migration.

### Engine and Pipeline historical notes retained for cutover

The following statements preserve minimum provenance from the former Engine
and Pipeline documents. They are dated historical records, not current status
or fresh verification. Current state is owned only by `handbook/STATUS.md`.

- The former Engine document presented PMLP-00 shared and external-OHE checks
  as acceptance criteria. Those criteria are now in the neural preprocessing
  contract and do not prove that a production artifact passed. It also recorded
  a snapshot claiming five Train-only preprocessing folds, 10 CSR `float32`
  matrices, 15 payload hashes, and PMLP-01 completion with 205,310 OOF rows.
  Direct artifact access was unavailable during the later reconciliation, so
  PMLP-00 remains `UNVERIFIED` in STATUS.
- The former Pipeline recorded the PlainMLP sequence from configuration review
  through full-Train fit, Validation thresholding, Test evaluation, and final
  review as `PMLP-02` through `PMLP-06`. STATUS is the authority for which of
  those stages currently have evidence.
- The former Pipeline recorded RealMLP stages `RMLP-01` through `RMLP-09`, with
  `RMLP-10` pending. Its dated CV snapshot reported mean ROC-AUC
  `0.8588438202`, standard deviation `0.0020634789`, OOF ROC-AUC
  `0.8588220609`, and a Train-only fixed-epoch decision of 51. Later full-Train,
  Validation, and Test state belongs to STATUS and the versioned artifacts.
- The former Pipeline also recorded a claim that 42 related RealMLP tests had
  passed. No retained test invocation or result report was found during the
  documentation reconciliation. The claim remains `UNVERIFIED`; this absence
  does not establish that the tests failed.

## G2 — canonical-boundary review

The rewrite retained intentional interface overlap while assigning one owner
for each responsibility:

| Owner | Canonical responsibility |
| --- | --- |
| Dataset B Data Selection / data contract | Cohort, target, column roles, audit metadata, exclusions, candidate universe, and stable data gates. |
| Feature Selection / feature governance | Evidence and decisions within the candidate universe, including priority screening and unresolved blocks. |
| Dataset B Pipeline / split and evaluation | Execution flow, split/CV/refit/Validation/Test boundaries, and evaluation contract. |
| Dataset B Engine Preprocessing / preprocessing contracts | Classical and neural preprocessing, external OHE and native adapter boundaries. |
| Neural Training Guideline / neural runbook | Neural runtime, pilot, history, atomic publication, recovery, and handoff procedure. |

Decision locators in the controlled rewrite:

- Temporary training feature lock and feature decisions: Feature Selection,
  section 13.
- Split, CV, configuration selection, refit, Validation, and Test boundaries:
  Dataset B Pipeline, section 5.2, and the split/evaluation contract.
- Classical/neural preprocessing and engine input routes: Dataset B Engine
  Preprocessing and the preprocessing contracts.
- Neural runtime, pilot, history/visuals, atomic fold publication, recovery,
  and TabMmini opening conditions: Neural Training Guideline and neural
  runbook.
- RQ1 metrics, bootstrap, and uncertainty: Project Planning Recover and the Set
  B Technical Requirement DOCX files.

Two distinctions were preserved rather than inferred away:

1. The priority-screening result is 12 `keep`, one `exclude_from_main`, and two
   representation-comparison features. It is not a final main-model feature
   lock while representation and other candidate decisions remain open.
2. Data Selection's handoff of future grouped validation and decision rules to
   Feature Selection does not claim that full model-based validation,
   six-model training, GSV/SL, or fairness evaluation has run.

## G3 — controlled rewrite

The former README did not contain a section titled G3 or a G3-specific
checklist. The migration map and acceptance text document the controlled
rewrite at a high level, but this archive does not invent missing G3 steps or
claim additional verification beyond the recorded G1, G2, and G4 content.

## G4 — acceptance checklist

The recorded acceptance checks were:

1. Preserve the `35 + 9 + 1 + 54 = 99` role count and all audit columns.
2. Keep raw `action_taken` outside `X` and preserve cohort and
   `y_approved` mapping.
3. Keep 12/1/2 screening statuses and explicitly open decisions distinct.
4. Retain every former heading group or give it an explicit replacement link.
5. Resolve Markdown links and referenced notebook/config paths.
6. Leave the three DOCX reports unrenamed, unremoved, and unedited.
7. Limit the documented migration to authorized Markdown files and avoid
   claiming completion for unrun models, GSV/SL, or fairness metrics.

## G4 — reference checks recorded at migration time

The source README recorded that:

- both source files referenced the current Set B Revised DOCX;
- Feature Selection consistently referenced
  `notebooks/eda_datasetB/05_feature_selction.ipynb`;
- path existence was checked, while the notebook itself was not rerun by the
  documentation rewrite.

These statements preserve what the former README recorded; they are not a
fresh check of current paths or current execution evidence. The notebook
reference statement is superseded by the provenance correction in G1: notebook
`04` owns the saved priority-screening evidence, while notebook `05` owns the
saved review-candidate checkpoint evidence.
