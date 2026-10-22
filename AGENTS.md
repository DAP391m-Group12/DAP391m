# DAP391m Project Guidance

## Purpose

Keep repository context selective. Read only the sources needed for the current
task; do not preload the full handbook, notebooks, artifacts, or reports.

## Project boundaries

- Dataset A is the bridge-analysis branch. Dataset B is the main experimental
  branch. Do not transfer fitted preprocessors, feature matrices, splits,
  models, predictions, or results between them.
- The project predicts mortgage application approval for the documented HMDA
  cohort. Do not reinterpret outputs as default risk, creditworthiness, or
  causal or legal evidence of discrimination.
- Treat raw data as immutable. Do not overwrite source data.
- Do not add models, features, metrics, experiments, dependencies, or pipeline
  stages unless the user explicitly requests them.

## PlainMLP closure guardrail

- Treat PMLP-05 artifacts as frozen. Do not retrain, rerun inference, regenerate
  bootstrap indices, or overwrite PMLP-05 artifacts unless the user explicitly
  opens a new task that authorizes that work.
- Treat PMLP-06 as closed and complete while its inputs, metric implementation,
  and approved numerical-comparison policy remain unchanged. Do not reopen or
  rerun PMLP-05/PMLP-06 merely because an older checklist, handoff, or review
  mentioned stricter byte-level reproducibility.
- Use exact SHA-256 only for stored-artifact integrity, lineage, identifiers,
  predictions, thresholds, and locked bootstrap indices. Do not use raw byte
  identity or a canonical digest as the numerical-equivalence PASS gate for a
  recomputed bootstrap table.
- For PMLP-06 bootstrap recomputation, compare replicate identifiers and
  valid/invalid payloads exactly; compare `calibration_slope` after rounding
  stored and recomputed values to six decimal places; compare the other nine
  metrics and all confidence intervals with the locked tolerance policy.
- A later explicit user decision supersedes conflicting older notes. Do not
  revive a superseded SHA, byte-identity, or canonical-digest requirement
  without a new explicit user instruction.
- The next model-development stream after the completed PlainMLP closure is
  TabMmini under its own separately authorized task contract. Do not modify
  PlainMLP as part of that work unless its Write Scope explicitly includes it.

## Context routing

- Use `README.md` only when the task needs project-wide architecture or the
  Dataset A/Dataset B boundary.
- For Dataset B documentation, start with `docs/handbook/README.md`. It is the
  documentation router; follow only the route relevant to the task.
- Read `docs/handbook/STATUS.md` only when the task depends on current execution
  state, completed work, open gates, or runtime evidence.
- Use the data contract for cohort, target, schema, audit roles, exclusions, or
  leakage questions.
- Use feature governance for feature evidence, representation decisions, or
  feature-lock questions.
- Use the appropriate preprocessing contract for classical or neural input
  handling.
- Use the split and evaluation contract for split membership, CV, thresholding,
  metrics, bootstrap, or RQ gates.
- Use a training runbook only for the model family being trained, validated,
  recovered, or handed off.
- Files under `docs/handbook/archive/` are historical and are not part of the
  default reading path.

## Soft context budget

- Start with the documentation router, the single directly applicable contract
  or runbook, and the target implementation and test files named by the task.
- Do not load adjacent components merely because they may be related. Expand
  the read scope only when the initial evidence is insufficient to complete or
  validate the task reliably.
- Before expanding, state the specific missing information and why the next
  file or component is needed. This is a progress note, not a request for user
  approval; continue unless the expansion would change the authorized task or
  write scope.
- When expanding, open the smallest relevant source first. Stop gathering
  context once the task can be completed and validated without speculation.
- This is a relevance budget, not a fixed file or token limit. Never omit
  evidence required for correctness solely to keep the context small.

## Large-file discipline

- Do not read notebooks as raw JSON in bulk. Inspect only the named notebook and
  the cells or outputs required by the task.
- Do not execute notebook cells, restart kernels, or run all cells unless the
  user explicitly authorizes execution.
- Do not recursively inspect `artifacts/`, `data/`, `logs/`, or generated report
  outputs. Open only paths directly required for the task.
- Prefer targeted file search and focused source inspection over repository-wide
  content dumps.

## Codex handover and chat approval

- Codex owns current RQ4b implementation, testing, review and reporting. Claude
  has no active implementer/verifier role or pending-work dependency. Preserve
  historical authorship and independent-review evidence without relabeling it.
- Approvals and acceptance for this handover are given in chat, not through
  terminal Allow requests. Commands may execute approved tests; do not use the
  terminal as an approval channel. Report any system permission barrier in chat.
- Work only within the approved task contract and Write Scope. An explicit
  approval of the full sequence permits progression only after each gate PASS;
  an approval of one task does not open the next task automatically.
- Complete implementation, then a separate self-review and validation pass.
  Do not label Codex checking its own work as independent review.
- Record only evidence-backed acceptance and limitations. STATUS remains the
  sole current execution register. Preserve locked method decisions.
- Handover approval does not authorize production preflight, training, real
  inference/probe fitting, new experiments, or changes to frozen artifacts.

## Changes and validation

- Preserve existing architecture and make the smallest change that satisfies
  the authorized task.
- Do not modify unrelated files or clean up pre-existing working-tree changes.
- Validate implementation with the narrowest relevant tests and inspect the
  resulting diff. Report any validation that was not run or could not complete.
