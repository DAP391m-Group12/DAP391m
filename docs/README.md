# Dataset B — Documentation Guide and Migration Map

## Purpose and current authority

This page is the entry point for Dataset B documentation. It records document
ownership and the completed migration map. It is intentionally short so that a
working session can open only the document needed for its task.

After the controlled rewrite, three Markdown files are authoritative for
separate responsibilities:

- [Dataset B Data Selection](Dataset_B_Data_Selection.md) owns the Dataset B
  contract: cohort and target, the 99-column role partition, candidate and
  exclusion inventory, audit metadata, and the mandatory data gates.
- [Feature Selection](Feature_selection.md) owns feature-governance evidence:
  the priority-feature screening, feature/block decisions, unresolved
  candidates, validation protocol, required notebook artifacts, and the
  temporary training feature lock.
- [Dataset B Pipeline](Dataset_B_Pipeline.md) owns execution modes,
  split/CV/refit/validation/test flow, train-only preprocessing boundaries,
  six-model routes, and artifact gates.

The technical-requirement DOCX files are retained as reports. This guide does
not replace or rename them.

## G1 — content migration map

The controlled rewrite is complete. The table records where each former
heading group is retained, condensed, or handed off; no factual evidence was
deleted.

| Current source and heading group | Controlled destination | Treatment in controlled rewrite |
| --- | --- | --- |
| `Dataset_B_Data_Selection.md` — 1. Mục tiêu và ranh giới diễn giải | Data contract | Retain cohort, target, interpretation limits, and the rule that transaction flags do not filter the cohort. |
| `Dataset_B_Data_Selection.md` — 2. Đối soát toàn bộ 99 cột | Data contract | Retain the `35 + 9 + 1 + 54 = 99` reconciliation and `record_id` boundary. |
| `Dataset_B_Data_Selection.md` — 3. Các cột được giữ (3.1–3.3, including A–C) | Data contract | Retain the target source, nine audit columns, 35-candidate inventory, and raw-versus-derived alternatives. Link to feature decisions; do not duplicate screening evidence. |
| `Dataset_B_Data_Selection.md` — 4. Các cột bị loại hoàn toàn (4.1–4.2) | Data contract | Retain the complete Gate 1–2 exclusion list and reasons. |
| `Dataset_B_Data_Selection.md` — 5. Các vấn đề phải kiểm tra (5.1–5.6) | Data contract | Retain mandatory schema, lineage, timing, applicability, and leakage gates; point to Feature Selection for completed evidence. |
| `Dataset_B_Data_Selection.md` — former 6. Thiết kế thực nghiệm, 7. Quy tắc quyết định | Feature-governance protocol | Retained in Feature Selection 10.1–12; Data Selection now contains a handoff link only. |
| `Dataset_B_Data_Selection.md` — former 8. Trạng thái hiện tại | Data-contract status | Retained as current section 7 with contract-level status only. |
| `Dataset_B_Data_Selection.md` — former 9. Nguồn tham chiếu | Source register | Retained as current section 8; internal paths were resolved during G4. |
| `Feature_selection.md` — 1. Mục tiêu, 2. Target, 3. Phân nhóm | Feature-governance introduction | Condensed to a Data Contract reference; role lists are not repeated. |
| `Feature_selection.md` — 4. Feature ưu tiên (4.1–4.6 and summary) | Feature-governance evidence | Retain all screening evidence and the 12 `keep` / 1 `exclude_from_main` / 2 representation-comparison statuses. |
| `Feature_selection.md` — 5. Feature cần kiểm tra (5.1–5.5 and checkpoint) | Feature-governance evidence | Retain candidate-block evidence, timing gates, and unresolved decisions. |
| `Feature_selection.md` — 6. Representation thay thế | Feature-governance decision gate | Retain the mutually exclusive raw-versus-derived comparisons. |
| `Feature_selection.md` — 7. Metadata (7.1–7.2) | Data-contract cross-reference | Keep diagnostic handling only; the canonical 9-column inventory remains in Data Selection. |
| `Feature_selection.md` — 8. Minimal audit, 9. Redundancy, 10. Sensitivity, 11. Decision rules | Feature-governance protocol | Retain methods and decision criteria; do not place model results here unless separately evidenced. |
| `Feature_selection.md` — 12. Notebook outputs | Execution contract | Retain required artifacts and the separation between screening and full training. |
| `Feature_selection.md` — 13. Decision state | Feature-governance status | Retain completed, provisional, and open decisions without converting them into a final model feature lock. |
| `Feature_selection.md` — 14. Related documents | Source register | Retained; internal paths were resolved during G4. |

All lower-level headings remain with their listed parent group unless the map
identifies an explicit Feature Selection handoff.

## G2 — canonical-boundary review

The overlap is intentional at the interface but should not be copied into a
third narrative document:

- **Data Selection** is the canonical source for the stable data boundary:
  target/cohort, column roles, audit metadata, exclusions, and the candidate
  universe.
- **Feature Selection** is the canonical source for evidence and decisions
  inside that candidate universe, including the completed priority screening
  and unresolved blocks.
- [Dataset B Pipeline](Dataset_B_Pipeline.md) contains execution modes and
  split/CV/refit/validation/test flow. It references—not repeats—the feature
  list, metrics, or role tables.

The single decision locators are:

- the temporary training feature lock and all feature decisions: Feature
  Selection, Section 13;
- split, `StratifiedKFold`, configuration selection, refit, validation and
  test boundaries: Dataset B Pipeline, Section 5.2;
- the RQ1 metric, bootstrap, and uncertainty contract: [Project Planning Recover](Project_planning_recover.docx) and [Set B Technical Requirement](<Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx>).

Two wording risks must be preserved as distinctions, not resolved by
assumption:

1. The priority-screening result is **12 `keep` features**, one
   `exclude_from_main` feature, and two representation-comparison features.
   It is not the final main-model feature lock while representation and
   remaining candidate decisions are open.
2. Data Selection section 6 now hands future grouped validation and decision
   rules to Feature Selection. The latter records completed screening and a
   future protocol; it does not claim that full model-based validation,
   six-model training, GSV/SL, or fairness evaluation has run.

## G4 acceptance checklist

G4 validates the controlled rewrite with the following checks:

1. The `35 + 9 + 1 + 54 = 99` role count and all audit columns remain intact.
2. Raw `action_taken` stays outside `X`; cohort and `y_approved` mapping are
   unchanged.
3. The 12/1/2 priority-screening statuses and all explicitly open decisions
   remain distinguishable.
4. Every old heading in the migration map has either retained content or an
   explicit replacement link.
5. Markdown links and referenced notebook/config paths resolve in the current
   repository.
6. No DOCX is renamed, removed, or edited by the Markdown consolidation.
7. The diff affects only authorized Markdown files and contains no claim that
   unrun models, GSV/SL, or fairness metrics have been completed.

## G4 reference status

The source Markdown references were resolved during G4:

- Both source files now reference the current Set B Revised DOCX.
- Feature Selection now references
  `notebooks/eda_datasetB/05_feature_selction.ipynb` consistently.

Path existence was checked during G4. The notebook itself was not rerun by the
documentation rewrite.
