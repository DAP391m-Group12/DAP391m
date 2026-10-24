# Dataset B neural training runbook

## 1. Purpose and ownership

**Current RQ4b state (2026-10-24):** RQ4b-3 is closed. All nine amended
development units are published and Gate B is `PASS`. Gate C aggregation
(`rq4b3_gatec_20261024_01`), the cross-run Test run
(`rq4b3_test_20261024_01`) and its package validation are complete, and
Notebook 12 reports the results; see STATUS for the evidence. The separate
evaluation-only applicant-sex supplementary run
(`rq4b3_sex_supplementary_20261024_02`) is reported in Notebook 13. Do not
rerun, reopen or mutate any source, Gate C, Test or supplementary run; no
final-fit stage was run, and no automatic retry is authorized. RQ4b-6 remains
deferred.

**Scoped chat override (2026-10-19):** the user authorized Codex to implement
terminal reports every 120 seconds, retain detailed durable file logging, test
and separately self-review, then execute one fresh real N4 probe, verify its
artifacts and stop before N5. This overrides user-only execution statements in
this runbook only for that sequence. No automatic retry or N5/production launch
is authorized. Implementation/execution is deferred to the next chat; consult
STATUS for actual progress and the interrupted `_02` attempt.

Terminal reporting must be periodic rather than printing every batch/epoch.
Report start, phase transitions, completion and errors immediately, even when
the operation finishes before 120 seconds. Keep detailed epoch/history and
failure evidence in files without throttling those records. Periodic reports
identify run/attempt, phase/fold/epoch/batch as applicable, elapsed time and the
last confirmed completed-progress time; observer activity is not optimizer
progress. Validate the reporting change before the single authorized real fit.

RQ4b uses the separate model-specific protocol in
[RQ4 mitigation plan, Section 6](../RQ4_MITIGATION_PLAN.md#6-rq4b-adversarial-realmlp-protocol)
and `config/dataset_b/rq4b_adversarial.yaml`. For every lambda/seed, five
Train-only folds each run all 256 epochs; the minimum unweighted mean
prediction CE selects the refit epoch, with exact ties selecting the earliest.
This is post-training selection, not patience-based early stopping or the
historical fixed-50 RealMLP pilot. Full-Train refit starts fresh without a
stopping holdout. The generic pilot branches below do not override this route.
Use `src.dataset_b.mitigation.rq4b_runner` for the separate staged RQ4b route;
legacy RealMLP commands do not implement that protocol. The spike uses synthetic
fixtures and at most two epochs per smoke fit; it does not execute production CV.
Codex owns current implementation, contracts/config, tests, self-review and
reporting. Approvals for the handover are given in chat, not terminal Allow
requests. Historical Claude implementation and Codex independent-review records
retain their authorship. Except for the current two-unit Claude Code assignment
above, the user executes real fits after the scoped readiness handoff. Codex supplies exact
commands, fixes scoped bugs, reports technical acceptance and checks saved
artifacts after each user run. Codex does not launch or retry real fits. There
is no Claude reviewer or verifier dependency for the current assignment.
Runtime/data checks and execution still need their own task scopes;
documentation approval is not production authorization.
The user locked production adversary `2 -> 32 ReLU -> 3`, no dropout/class
weighting/search, adversary seed equal to model seed with isolated RNG, and
refit schedule horizon 256 while executing only the selected epoch count.
Implementation and real development execution evidence are recorded in STATUS;
the nine-unit Gate B runtime scope is verified, but Gate C aggregation and
scientific mitigation support are not. The user has locked capability probe population,
60/20/20 split, AdamW budget, Y-only comparison and synthetic gate; the separate
probe uses patience 15, not predictor CV. Test uncertainty is locked-index
paired percentile bootstrap (1,000/1,000 valid), with strict strong-support CI
endpoints and sample SD (`ddof=1`). Final probe AdamW parameters, CE-based
stopping/checkpoint, stored-probability log-ratio, synthetic fixture and linear
percentile interpolation are locked in the RQ4 plan/YAML. Method decisions are
unchanged. Remaining integration verification and explicit user launch are
separate gates; §9 of the plan owns sequencing and §10 owns interpretation. Report stability as
`not_assessed` and distinguish scientific support from execution success.

### RQ4b stage operation and user-fit handoff

These instructions describe the current interface, not a launch approval or
runtime PASS. Use STATUS for execution state. Codex prepares and tests code,
fixes scoped bugs, performs a separate self-review/validation pass and reports
technical acceptance. The user runs every real probe/CV/refit command; Codex
checks outputs before recommending the next unit. Non-fit preflight,
inference and evaluation also require their own authorized task scopes.

Run from `D:\Proj\DAP\DAP391m_GR12` using the existing GPU environment
`.venv-realmlp-gpu\Scripts\python.exe`. Before command handoff, verify the
environment and pinned PyTabKit lineage against the locked configuration,
mandatory acceptances and a finite operation on `cuda:0`; no CPU fallback or
dependency installation is implied. Environment existence is not readiness.
For the amended RQ4b-3 execution, the frozen sources are Track A run
`rq4b3_adv_20261019_02` and seed-43 Track B run
`rq4b3_adv_20261021_01`. Seed 44 uses only the v4 continuation
`rq4b3_adv_20261022_01`; never resume either source run with current code.
Finish required code fixes before creating the run: `open_run` rejects changes
to any of the five recorded RQ4b source files. Do not edit manifests to bypass
that check. Every CLI stage requires `--confirm-production`; this flag is an
execution acknowledgement, not a replacement for the scoped chat decision.

The PowerShell command template is:

```powershell
& .\.venv-realmlp-gpu\Scripts\python.exe -m src.dataset_b.mitigation.rq4b_runner --stage <stage> --run-id <run_id> --confirm-production
```

For `development`, `development_validation`, `final_fit` and `final_validation`,
append `--lambda <lambda> --seed <seed>`. Other stages take neither option.
For `capability` only, supply `--real-probe-evidence <path>` to ingest an
already completed real-probe attempt. This never fits a probe. If omitted,
the real probe is explicitly recorded as not run; if supplied but missing,
invalid or failed, the stage fails rather than silently falling back.
These are templates: replace placeholders at handoff and run one explicit unit
at a time. There is no whole-grid command in this interface.

For the seed-44 continuation, the active protocol is
`rq4b_adversarial_realmlp_v4_seed44_continuation`. The runner rejects
production create/open requests outside `rq4b3_adv_20261022_01`, verifies the
pinned manifest and receipt digests of both source runs, and rejects a manifest
whose continuation lineage differs. This does not replace the fresh
per-run preflight and capability gates. Do not use the historical Cartesian-grid
selection or seeds 45--46 routes.
Gate C requires an evaluation-only aggregation route over all three immutable
source runs; no such production CLI command is approved by this runbook yet.

The machine-readable v4 assignment contains only `(0,44)` and `(0.01,44)`.
Both units completed in the required order with verified receipt/output hashes,
five-CV-plus-one-refit accounting and 43,995 Validation rows per unit. Source
runs remain read only and their fitted state must not be migrated or refitted.
The current v4 runner is not the cross-run aggregation route; use a new
evaluation-only run identity after its contract and implementation are approved.

Paths below are repository-relative. Let `R` be
`artifacts/dataset_b/rq4/rq4b/<run_id>`. Each published stage lives at
`R/stages/<key>/` with `receipt.json`; verify receipt identity, input lineage
and all output hashes before proceeding. Unit keys use
`<prefix>_l<lambda>_s<seed>`, replacing a decimal point by `p` (for example,
`dev_l0p01_s42`). Failures retained by the stage writer live under
`R/failures/<key>.<uuid>/` with `FAILURE_RECEIPT.json`. Receipt writing itself
can fail; keep the original exception, logs and any temporary directory.

| CLI `--stage` | Prerequisites and inputs | Published output/key | Real work and stop check |
|---|---|---|---|
| `create` | Separately approved readiness handoff; unused continuation run ID; locked YAML/current source; exact pinned hashes for both immutable source runs | `R/run_manifest.json`, empty stages/failures directories | No fit/inference. Stop on collision, invalid ID, wrong namespace or any source manifest/receipt digest mismatch. |
| `preflight` | Run manifest; T1-T5 acceptance records; spike/T1/source-lineage prerequisites; CUDA finite operation; locked index file and manifest presence | `preflight/preflight.json` | No fit/model inference. Stop unless every collected status is PASS. This stage does not validate all real input bodies or prove fit/reload readiness. |
| `capability` | Verified preflight; retained synthetic gate; optional explicit `--real-probe-evidence` path | `capability/capability.json`, including the validated real evidence and invocation when supplied | No fit/model inference. Ingestion reconstructs the frozen population and verifies input/code lineage, split, settings, comparator, fit history and bootstrap arithmetic. Stop on invalid evidence; inconclusive capability is not a stop gate. |
| `development` | Gate A PASS; preflight/capability receipts; Train split, feature contract, manifest and audit join; exactly one approved seed-44 unit; `(0.01,44)` additionally requires verified `dev_l0_s44` and `devval_l0_s44` receipts | `dev_l*_s*/`: preprocessor/model, CV curves, complete per-fold outputs, selection record, diagnostics, run log, JSONL/final history and fit ledger | A separately authorized operator runs 5 Train-only CV folds of 256 epochs and 1 fresh refit. Only `(0,44)` and `(0.01,44)` belong to v4; run the control first. Do not refit or reuse fitted state from seed 42/43. Stop on exception, nonfinite values, logging failure, invalid 5x256 curves, serialization/reload or lineage failure. |
| `development_validation` | Matching development fit receipt/files; complete Validation split and audit metadata | `devval_l*_s*/validation_predictions.csv`, `threshold.json` | Validation inference and global calibration; no fit. Stop on population, probability, metadata or artifact validation error. |
| `selection` | Gate C prerequisites: evaluation evidence for all nine retained/new units, saved thresholds, EOD3/paired delta EOD3/BA cost, EOD6 audit, seed variability, training stability/integrity and an explicitly approved three-seed selection policy | `selection/candidates.json`, `selection_lock.json` | No fit/new model inference. The existing Cartesian-grid implementation must not be used. Stop while the revised selection policy is pending; no Test-based selection. |
| `final_fit` | Not active in the amended nine-unit design unless a later approved selection policy explicitly requires a new fit | Historical v2 `final_l*_s*/` route | Do not run seeds 45--46 or add any lambda/seed. Any reactivation requires a separate contract amendment. |
| `final_validation` | Not active unless an authorized `final_fit` exists under a later amendment | Historical v2 `finalval_l*_s*/` route | Do not run under the current amended contract. |
| `test_evaluation` | Gate C PASS; approved selection lock; verified retained/new arms for seeds 42--44; Test split/audit join; locked indices and manifest | `test_evaluation/`: deduplicated Test predictions, `seed_42.json` through `seed_44.json`, `assessment.json`, `status.json` | Test inference and paired bootstrap; no fit. Stop on missing policy/threshold locks, lineage/ID/index mismatch or runtime error. A valid criterion failure is reported, not retried to obtain a better result. |
| `package_validation` | Published Test evaluation and all published receipts/saved reporting inputs | `package_validation/package_report.json` | No fit/new model inference. Check hashes, lineage, the amended 45-CV/9-refit maximum budget and reporting traces; stop if any package check fails. |

The loader resolves Train/Validation/Test from
`data/dataset_b/split_v1/{train,validation,test}.csv`, with
`artifacts/dataset_b/split_v1.json`, the feature contract and authorized raw
derived-race reconciliation source through `rq4b_inputs.load_contract()`.
Use only resources permitted by the stage; do not replace them with cached
matrices from another branch/run. Test indices are
`artifacts/dataset_b/rq1_test_bootstrap_indices.npz` with
`rq1_test_bootstrap_manifest.json` beside it; do not regenerate them.

**Real probe is a separate user-fit operation.** Its frozen source is
`artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/validation/validation_predictions.csv`
plus authorized race reconciliation. No baseline refit or new frozen-model
inference is permitted. After the separately scoped runtime/readiness checks,
the user executes this template with a new attempt name:

```powershell
& .\.venv-realmlp-gpu\Scripts\python.exe -m src.dataset_b.mitigation.rq4b_probe --kind real --attempt-name <new_attempt_name> --confirm-real-probe
```

The command creates one direct child of `scratch/rq4b_implementation/`, loads
the frozen population through the contract and fits once. Successful output is
`invocation.json` plus immutable `real_probe_evidence.json`; preparation/fit/
publication failures retain `FAILURE_RECEIPT.json` when receipt writing succeeds.
Never reuse an attempt. Omitting `--kind real` preserves the synthetic-gate
CLI; `--confirm-real-probe` is required for real mode and rejected for synthetic.
This acknowledgement does not authorize Codex to run real fits.

Before publishing the capability stage, ingest the saved result with:

```powershell
& .\.venv-realmlp-gpu\Scripts\python.exe -m src.dataset_b.mitigation.rq4b_runner --stage capability --run-id <run_id> --confirm-production --real-probe-evidence scratch/rq4b_implementation/<new_attempt_name>/real_probe_evidence.json
```

Do not publish a placeholder capability stage and then try to overwrite it.
Evidence must come from the frozen-source CLI route, not a synthetic or legacy
in-memory attempt. Ingestion checks source content hashes, producing probe/input
code, reconstructed population and saved invocation; it includes the evidence
and invocation in the stage receipt's hashed payload. It rereads frozen inputs
but performs no fitting or model inference. Keep inputs and producing code
unchanged between fitting and ingestion. These checks establish lineage and
internal consistency, not independent proof of GPU execution or fitted-model
correctness; Codex still checks the user's logs/environment after the run.
The report records real execution as `completed` with gain and CI, rather than
`not_run`. It conservatively retains `capability_inconclusive`: the locked real
probe is descriptive and has no automatic binary promotion rule here. It does
not apply the synthetic gain cutoff to the real probe. Capability inconclusive
is not a new production PASS/FAIL gate. Consistent bootstrap evidence with
`ci_evaluable=false` is retained descriptively; forged validity counts or CI
values are rejected. A runtime failure still requires
investigation. A missing/failed/corrupt explicitly supplied attempt is an input
error, not a negative scientific finding.

**Failure and recovery:**

- Never overwrite a published stage, delete a failure to unlock a key, resume
  fitted state or transplant stages into another run. Inspect the exception,
  logs, receipt and partial outputs first; failures before the writer boundary
  may have no receipt and must not be described as successful attempts.
- Only `development` and `final_fit` forward
  `--new-attempt-after-failure`. After Codex diagnoses the failure, a scoped
  user-executed fresh attempt may use the same run/unit if code lineage is
  unchanged and the unit is unpublished. It reruns the whole unit; this is not
  checkpoint resume or automatic retry.
- The flag does not enable retry for preflight, capability, Validation,
  selection, Test evaluation or package validation. A retained failure for
  those keys blocks that stage in the same run. Stop and plan recovery in chat;
  a new run entails fresh prerequisites/work and explicit accounting, not reuse
  of the previous run's stage directories. Do not silently redesign retry.
- An interrupted/partial `create` also must not be reused by deleting its
  directory. A real-probe attempt is single-use; retain it and use a new
  explicitly scoped attempt after diagnosis. If source changes are required,
  the existing production run cannot continue across the code-lineage change.

**Fit accounting and post-run review:** the amended development design has nine
units, not a Cartesian product. Retain the five completed seed-42 units at
lambdas `[0, 0.01, 0.03, 0.1, 0.3]`. Track B may add only `(0,43)`,
`(0.01,43)`, `(0,44)` and `(0.01,44)`. Lambda 1 and seeds 45--46 are outside
the execution queue. The amended maximum is 45 CV fold fits plus nine refits,
including the five retained units; Track B itself accounts for at most 20 CV
fold fits and four refits. Do not interpret the maximum as permission to refit
retained units.
Package counts are per-published-unit declarations, not a complete measure of
all attempted compute. Keep a separate handoff/audit ledger for probe fits,
spike, failed/partial fits and retries, including run/attempt IDs, unit, command,
timestamps, exit status and observed completed/started fits. Mark unknown
partial counts unknown rather than zero; no retry allowance is automatic.
Codex checks receipts, curves, epoch selection, reload diagnostics, predictions,
thresholds and lineage after user execution, then reports whether the next
stage's prerequisites are met. Preserve execution, statistical evidence,
fairness effectiveness, predictive utility, capability and training stability
as separate conclusions. The old 4/5 strong-support rule is not satisfied by
three seeds, and no 2/3 replacement is approved.

This runbook is the operator-facing procedure for Dataset B neural training. It
owns:

- the GPU and runtime gate;
- Train-only pilot execution;
- training history and local observability;
- the decision between a fixed epoch budget and Train-only inner stopping;
- atomic artifact publication and crash recovery;
- the handoff between implementation, execution, Validation, and Test stages;
- the distinction between internal checks and gates that need user approval.

It does not own feature selection, split membership, preprocessing semantics,
model results, or current execution status. Use:

- [Feature governance](../contracts/feature_governance.md) for feature decisions;
- [Split and evaluation](../contracts/split_and_evaluation.md) for partition and
  evaluation boundaries;
- [Neural preprocessing](../contracts/preprocessing_neural.md) for model input
  routes and fitted preprocessing state.

The concrete commands in this runbook are verified against the current RealMLP
interfaces. PlainMLP and TabMmini have their own locked configurations and
model-specific interfaces; a future neural model may reuse this process only
after equivalent contracts exist. Parameters must not be copied between model
families by implication.

TabMmini v1's approved model-specific decisions are locked in the
[TabMmini v1 training runbook](tabmmini_training.md). Its reusable source and
final run are now closed; consult [STATUS](../STATUS.md) for the authoritative
execution evidence rather than treating this procedural runbook as a status
register.

PlainMLP's final policy is locked in
[`plain_mlp_v1.yaml`](../../../config/dataset_b/plain_mlp_v1.yaml). Its source
and final run are closed; [STATUS](../STATUS.md) owns the execution evidence.
Any later command still requires an explicitly authorized new run and must not
resume the frozen fitted state.

## 2. Authorization envelope and gate types

Before changing code or running a stage, record the authorized model, files,
partitions, decisions, execution stages, and validation activities. Apply these
gate types consistently:

| Gate type | Examples | User approval rule |
| --- | --- | --- |
| Internal verification | Static review, unit tests, manifest/schema/content validation, dtype checks, and no-training CLI checks | Does not create a new authorization gate. Run it whenever it is required inside an already authorized scope. |
| Operational safety | GPU/CUDA probe, backend import, finite tensor smoke, history smoke, atomic-rename probe, disk/runtime checks | Does not require separate approval when execution is already authorized. A failure stops the run. |
| Data boundary | First access to project Validation; first access to Test | Requires explicit authorization unless the active authorization already names that partition and stage. Validation authorization never implies Test authorization. |
| Research decision | Change feature set, model configuration, preprocessing policy, epoch/stopping policy, threshold policy, or evaluation policy | Requires explicit approval unless the exact decision was already granted in the active scope. Produce evidence first; do not implement an unapproved choice. |

One authorization covers the complete implementation–execution–validation
workflow that it explicitly grants. Do not pause for a second approval merely
because an internal test passed, a file was written, a stage completed, or a
reviewer wants to inspect the same in-scope evidence. Ask again only when the
next action crosses the recorded data boundary, changes a research decision, or
otherwise expands the authorized scope.

Visual QA is evidence for readability and diagnosis. By default it does not
block model execution and does not create an authorization gate. It becomes
blocking only when the active task explicitly makes a visual an acceptance
criterion, or when the visual exposes an underlying machine-readable artifact
failure that is itself blocking.

## 3. Required execution order

Use this order; stop at the first failed gate:

1. Confirm the authorization envelope and unchanged feature, split,
   preprocessing, and model configuration fields.
2. Create one new `run_id` and its empty run-scoped working/final destinations.
3. Run the final data-quality gate and stop before fitted state is created if
   any check fails.
4. Regenerate and validate the 15/19/62 preprocessing inputs for that `run_id`.
5. Run internal no-training checks.
6. Run the GPU preflight and compatibility/history smoke.
7. Materialize or validate immutable Train-only pilot IDs.
8. Run the pilot without project Validation or Test access.
9. Review pilot evidence and lock exactly one epoch/stopping policy. For
   PlainMLP, verify the already locked inner-stopping policy without changing
   it; the pilot is not a new configuration-selection gate.
10. Run a one-fold orchestration and atomic-publication smoke when the runtime is
   new or the publication path changed.
11. Run Train-only cross-validation with fresh fitted state per fold.
12. Reinitialize preprocessing and model state, then fit full Train using the
   locked policy.
13. Open Validation only under its data-boundary authorization; generate one
    ordered prediction artifact and select the threshold under the evaluation
    contract.
14. Open Test only under its separate data-boundary authorization and only
    after the model, preprocessing, and threshold are frozen.
15. Validate the complete bundle directly, mark the run `COMPLETE`, publish its
    final directory, validate it again, and freeze it against overwrite.

Pilot and smoke outputs are not CV, Validation, Test, or RQ1 evidence. Outer
fold-validation may be predicted once after a fold fit, but it must not control
the epoch budget, stopping rule, preprocessing, or backend-fitted state.

## 4. Pre-run checks

From the repository root:

1. Create one new `run_id` for the authorized final retraining execution and
   record it in every preprocessing, model, prediction, threshold, and
   evaluation manifest.
2. Use run-scoped directories such as
   `artifacts/dataset_b/<model>/working/<run_id>/` and
   `artifacts/dataset_b/<model>/final/<run_id>/`; confirm that neither target
   will overwrite or resume an earlier run.
3. Confirm the active feature, model, runtime, split, and preprocessing files
   by reading their parsed schema, types, values, and ordered lists directly.
4. Confirm the immutable `split_v1` membership, order, and row counts; do not
   regenerate it from a model runner.
5. Confirm that the selected route consumes the final representation assigned
   by the preprocessing contract: 15 raw inputs, 19 semantic columns, and 62
   PlainMLP OHE columns where external OHE applies.
6. Regenerate preprocessing inputs for the new `run_id`. Do not resume,
   extend, relabel, or consume a fitted preprocessor, native frame, OHE matrix,
   model, optimizer, or checkpoint from a 12-feature run.
7. Run the relevant unit and no-training tests. A test that imports only a CPU
   build is not GPU evidence.
8. Ensure the working destination is on the filesystem on which the final
   same-filesystem rename will occur.
9. Check free disk space for the temporary directory, model, histories,
   predictions, and retained failed attempts.
10. Confirm that no final output directory will be overwritten.

Record failures in the handoff. Do not repair a feature, config, or policy
difference as incidental cleanup; route it through the research-decision gate.

### 4.1 Mandatory data-quality gate

Run this gate before every pilot, CV execution, and full-Train fit. The gate
must validate content directly and record its evidence under the current
`run_id`:

1. the raw input contains the exact ordered 15-feature schema;
2. target, `record_id`, and audit-only columns remain outside model input;
3. no retained final input is null, empty, whitespace-only, or textual `nan`;
4. every raw token satisfies its field domain and numeric validity rules;
5. for each of the five exemption-coded flags,
   `raw_1111_count == normalized_exempt_count`;
6. `semantic_1111_count == 0` for every model-facing input;
7. normalization preserves retained row count, ordered `record_id`, target,
   and partition alignment;
8. the shared semantic frame has the exact ordered 19-column schema;
9. PlainMLP OHE has the exact ordered 62-column schema, while RealMLP and
   TabMmini receive the exact native 19-column schema;
10. category lists contain semantic `Exempt` where permitted and contain no
    model-facing `1111` or `MISSING` category;
11. dtype, sparse/native container, transformation, model parameter, and
    runtime fields match the approved route directly.

For PlainMLP, item 11 specifically requires fit-local `RobustScaler` on the
four numeric value columns only, CSR and current dense batch `float64`, and
`float64` parameters, targets, logits, and loss. Any 66-column, `float32`,
CPU-only, no-scaler, or fixed-100-epoch execution path fails this gate.

Any failed check returns `FAIL` and stops the run before preprocessing or model
fit. The operator may train only after this gate returns `PASS` for the current
stage and `run_id`. If an existing command or validator requires legacy
12-feature inputs or cannot produce this evidence, stop; do not use that path
for final retraining.

### 4.2 Schema and content validation

Feature, model, and runtime configuration files are accepted from their parsed
schema, types, values, ordered lists, and data-quality invariants. Manifests
must store the directly reviewable input schema, semantic schema, output
schema/category maps, transformation settings, model parameters, runtime
settings, row counts, ordered identifiers, and observed validation results.

Persisted artifacts are accepted by reopening them and directly checking their
expected container, shape, dtype, finiteness, value ranges, ordered
`record_id`, labels where permitted, prediction values, threshold, histories,
and manifest fields. Presentation-only YAML differences do not fail a run;
changed semantic content does.

## 5. GPU compatibility gate

Run a read-only preflight before writing runtime evidence:

```console
python -m src.dataset_b.runtime.run_realmlp_smoke --preflight-only
```

The preflight must identify Python, PyTorch, CUDA build/runtime, NVIDIA driver,
GPU and device index, PyTabKit version, and whether a finite CUDA tensor
operation succeeds. `torch.cuda.is_available()` must be true. A CPU result or
silent CPU fallback is a failure, not a degraded pass.

With execution authorized, publish the compatibility and two-epoch history
smoke:

```console
python -m src.dataset_b.runtime.run_realmlp_smoke
```

Accept the gate only when the runtime manifest, GPU log, smoke result,
two-column finite probabilities, and complete epoch records validate. Freeze
the selected device and batch size for the pilot. If an OOM occurs, use only
the fallback sequence implemented by the runtime gate and record every
attempt; do not change other model parameters opportunistically.

A runtime PASS is bound to its interpreter and environment fingerprint. Rerun
the gate after changing the interpreter, backend, GPU/CUDA environment, device,
or relevant runtime configuration.

For PlainMLP, the gate additionally requires device index 0, a finite CUDA
`float64` calculation, `float64` CSR-to-dense batch transfer, `float64` model
parameters, targets, logits, and loss, a successful forward/backward/optimizer
step, checkpoint reload with dtype preserved, and recorded peak memory. Batch
fallback may follow only a confirmed CUDA OOM and only in the locked sequence
256, 128, 64. Unavailable CUDA, any silent CPU fallback, or any model-facing
`float32` value returns `FAIL`/`BLOCKED`; it never authorizes a degraded run.

## 6. Pilot procedure

Pilot data must come only from immutable Train membership. The pilot-fit and
pilot-inner-stop ID files must be disjoint, reproduce their directly recorded
ordered membership, and contain no project Validation or Test IDs. The final
runner persists both ordered ID files inside the current run-scoped pilot
directory and validates their uniqueness, disjointness, and prediction order.

Run the RealMLP pilot through its public CLI:

```console
python -m src.dataset_b.runtime.realmlp_pilot_run run --run-id <run_id>
```

The pilot must:

- use the frozen pilot configuration and runtime-gate batch size;
- leave backend early stopping disabled while collecting the complete planned
  trajectory;
- use only the explicit Train-derived inner-stop set for loss history and
  post-fit pilot predictions;
- write live JSONL history after each completed epoch;
- finish in an `AWAITING_DECISION` working directory rather than publishing a
  final decision automatically.

Do not use the outer fold-validation, project Validation, or Test partitions to
choose an epoch or patience rule.

## 7. Fixed epoch versus inner stopping

Review the observed pilot history and choose exactly one policy:

### Fixed epoch

Choose `FIXED_EPOCH` only when the observed Train-only pilot trajectory
supports a defensible epoch budget. Record the selected epoch, rationale,
directly referenced history records, and the same full-Train refit policy. CV
folds and full-Train fit must use the locked budget consistently, must not
create an inner-stop set, and must represent inner-validation and best-epoch
history fields as not applicable.

### Train-only inner stopping

Choose `INNER_EARLY_STOPPING` only when the backend-supported procedure is
fully specified. Record the inner split fraction and seed, patience parameters,
metric, maximum budget, refit behavior, and directly referenced evidence. Each
fold must derive the stopping set from fold-train only. Full-Train refit must
derive any stopping set from Train only. Project Validation is never an
early-stopping channel.

If the curve remains unstable or the optimum is unresolved at the budget
boundary, do not guess. Propose another Train-only pilot budget or approved seed
as a research-decision change.

The `decide` command mutates the policy and publishes the pilot, so run it only
after the chosen policy is authorized or when that exact choice is already
inside the active authorization:

```console
python -m src.dataset_b.runtime.realmlp_pilot_run decide --run-id <run_id> --fixed-epoch <N> --rationale "<evidence-based rationale>"
python -m src.dataset_b.runtime.realmlp_pilot_run validate --run-id <run_id>
```

The current RealMLP production CV and full-Train runners implement only the
fixed-epoch branch. If a future pilot selects `INNER_EARLY_STOPPING`, stop;
do not use the fixed-epoch runners until a separately authorized implementation
has added that stopping behavior and matching validators.

### PlainMLP locked inner-stopping policy

PlainMLP does not choose between the two branches above. It uses stratified
Train-derived inner stopping with fraction 0.10 and random state 42, monitoring
`inner_stop_bce_loss` in `min` mode with maximum 100 epochs, patience 10, and
`min_delta = 0.00001`. Restore only the selected best epoch number, then
discard the temporary selector, optimizer, and selector-fitted preprocessing.
A fresh final estimator and fresh fit-local preprocessor must refit the entire
outer fold-train or full-Train population for exactly that epoch count.

Across five-fold CV and the full-Train stage, require five temporary CV epoch
selectors, five final outer-CV estimators, one temporary full-Train selector,
and one final full-Train estimator. This is 12 fresh model initializations; only
the six final estimators are model evidence. Outer fold-validation, project
Validation, and Test must not enter an inner-stop set.

## 8. History and logging

Local files are authoritative; network monitoring is optional. Every completed
epoch record must contain the observed epoch, training loss, learning rate,
epoch and elapsed time, GPU memory, status, and timestamp. When an explicit
inner-stop set exists, record its observed loss and best epoch. Under a
fixed-epoch fit, keep those fields null with an explicit not-applicable status;
never copy training loss into a validation field.

Required durable outputs are:

```text
run.log
training_history.jsonl
training_history.json
```

Prefer a public callback or public history API, followed by a backend-supported
logger. Terminal parsing is temporary fallback evidence only. Do not
monkey-patch private backend internals in a production run. The implementation
in
[`realmlp_history.py`](../../../src/dataset_b/runtime/realmlp_history.py) is
version-bound and must fail rather than fabricate unavailable fields.

History must be sufficient to render loss, learning-rate, timing, and GPU-memory
diagnostics. Rendered figures help review but do not replace JSON/JSONL evidence
or block execution by default.

For RQ4b fit units, `run.log` and `training_history.jsonl` are append-only live
evidence in the stage attempt directory. Each write is flushed and fsynced.
Records identify UTC timestamp, run, attempt, unit, lambda, seed, phase and fold
when applicable. Completed-epoch records contain the observed batch count,
training loss, fold-validation CE when applicable, learning rate, epoch/elapsed
time and allocated GPU memory. A non-applicable value is null with a reason; it
must not be inferred or copied from another metric. Batch/time heartbeats record
the last callback position and explicitly do not claim that an optimizer step
completed.

`fit_ledger.json` records planned, started, completed, partial and unknown fit
counts plus the last observed phase/fold/epoch/batch and timestamp. Failure
receipts preserve the exception chain, last position, ledger accounting and all
already-flushed logs. A missing backend ledger after dispatch is `unknown`, not
zero compute. Any logging write failure is a stage failure and is not silently
ignored. On successful completion, `training_history.json` is finalized and
`cv_fold_outputs.json` retains every fold's complete history/log alongside the
curve and selected-epoch link; `cv_curves.json` alone is not sufficient.

Real capability-probe attempts use the same three log/history filenames inside
their single-use attempt directory. Their heartbeats have no production run ID
and state that reason. Probe failure receipts retain traceback and last progress;
the flushed log remains evidence after interruption.

For N4, the observer starts before frozen-population loading and source hashing.
Its time-based `observer_heartbeat` can report during preparation, a long batch,
validation, holdout, bootstrap and lineage verification when the interpreter
and filesystem can run. It does not claim optimizer progress. `timestamp_utc`
is observation time; `last_progress_utc` changes only on a recorded completed
epoch (and completed fit phase for the training ledger), never on a heartbeat.
Do not infer that an optimizer is moving merely because the observer is alive.
Hard process termination, an interpreter-wide stall or blocked storage can
prevent observations; no log-based liveness guarantee covers those cases.

Initial log/invocation writes are inside the failure-receipt boundary. Observer
write errors are reported on stderr and propagated at the next foreground log
check or observer shutdown; a blocked foreground GPU call cannot be forcibly
interrupted by the observer. No automatic retry is permitted.

Before N4 publication, stop the observer, append `computation_completed`, then
seal the three log/history files. This event is not an attempt-success claim.
The atomically published `real_probe_evidence.json` is the completion marker;
only the subsequent CLI summary reports `completed`. A failure while publishing
evidence leaves failure evidence even if computation and log sealing finished.
The evidence binds SHA-256 hashes of `run.log`, `training_history.jsonl` and
`training_history.json`. Ingestion requires all three, checks their integrity,
identity and event consistency, and compares every logged epoch's metrics to
the scientific history. Missing logs or mismatched histories fail validation.
Older N4 attempts remain immutable under their producing lineage; do not
rewrite them or relax current lineage/logging gates to reuse them.

## 9. CV and full-Train execution

After the data-quality gate, regenerated preprocessing inputs, pilot decision,
and runtime gate validate for the same `run_id`, the current RealMLP interfaces
are:

```console
python -m src.dataset_b.runtime.realmlp_cv run --run-id <run_id>
python -m src.dataset_b.runtime.realmlp_cv validate --run-id <run_id>
python -m src.dataset_b.runtime.realmlp_full_train run --run-id <run_id>
python -m src.dataset_b.runtime.realmlp_full_train validate --run-id <run_id>
```

Each CV fold must initialize fresh preprocessing, model, optimizer, and
backend-fitted state. Publish ordered fold predictions once after training;
aggregate OOF evidence only after every required fold validates. Full-Train
must also start from fresh state and consume only Train membership. A completed
CV model must not be reused as the full-Train model. Legacy 12-feature fitted
state is never a valid starting point. Rerun the data-quality gate immediately
before full-Train fit and stop if its directly recorded result is not `PASS`.

PlainMLP additionally recalculates `pos_weight = n_denied / n_approved` from
each population actually being fitted: inner-fit for a selector, complete
outer fold-train for its fresh final estimator, full-Train inner-fit for the
full-Train selector, and all 205,310 Train rows for the final full-Train
estimator. OOF evidence must cover every Train row exactly once. Each final
outer fold-validation prediction is generated once; selector predictions are
not OOF evidence.

## 10. Atomic publication and crash recovery

Write a run or fold under `working/<run_id>` on the same filesystem as its
`final/<run_id>` destination. During execution, the working directory is
diagnostic state only and must not feed OOF aggregation or downstream
evaluation.

Before publication:

1. finish training and the stage-appropriate prediction;
2. verify row count, exact ordered `record_id` membership, probability shape,
   finiteness, and range;
3. reopen and verify model, history, predictions, result, schema, configuration,
   normalization counts, equality gate, and manifest contents directly;
4. confirm every component records the same `run_id`;
5. set the manifest state to `COMPLETE` last and close open file handles;
6. validate the complete working bundle;
7. rename the whole `working/<run_id>` directory to `final/<run_id>` on the
   same filesystem;
8. validate the final directory again, including reporting and `run_id`
   lineage;
9. create `FROZEN` only after that post-move validation returns `PASS`.

A `COMPLETE` state inside a working directory does not make it published. A
model file alone is never completion evidence. Never create `FROZEN` in the
working directory or before post-move validation. Rename failure must preserve
diagnosable working state without a marker; post-move validation failure must
leave the final directory unmarked and must not report publication success.

The retained RealMLP and TabMmini final artifacts were directly reviewed and
frozen as historical execution evidence. Their reusable publication modules
now enforce the canonical working-review `PASS` -> atomic rename -> post-move
final validation `PASS` -> `FROZEN` order. This source guarantee does not
rewrite or restage the already frozen artifact directories.

On restart:

- never resume a legacy preprocessor, model, optimizer, or checkpoint;
- preserve an interrupted working directory as a numbered failed attempt;
- start the failed stage in a clean directory with fresh fitted state instead
  of mixing attempts;
- a completed fold from the same current `run_id` may contribute its validated
  predictions to OOF aggregation, but its fitted state must not initialize any
  other fold or the full-Train model;
- never overwrite a valid published directory.

## 11. Validation and Test handoff

The following commands exist for the current RealMLP evaluation path, but they
must be run only after the corresponding data-boundary authorization:

```console
python -m src.dataset_b.evaluation.realmlp_validation run --run-id <run_id>
python -m src.dataset_b.evaluation.realmlp_validation validate --run-id <run_id>
python -m src.dataset_b.evaluation.realmlp_validation_threshold run --run-id <run_id>
python -m src.dataset_b.evaluation.realmlp_validation_threshold validate --run-id <run_id>
python -m src.dataset_b.evaluation.realmlp_test_evaluation run --run-id <run_id>
python -m src.dataset_b.evaluation.realmlp_test_evaluation validate --run-id <run_id>
```

Before opening Validation, hand off the validated full-Train manifest and
fitted-model location, `run_id`, directly recorded split/config/preprocessing
fields, ordered Validation membership, probability convention, and confirmation
that fitting is complete.
Threshold selection may inspect Validation labels only for the approved
objective and may not refit preprocessing or model state.

Before opening Test, hand off the validated Validation prediction and threshold
artifacts, the same `run_id`, frozen model/preprocessing/configuration fields,
ordered Test membership, and bootstrap replicate identifiers. Test execution
must not fit, change the threshold, regenerate shared bootstrap membership, or
reopen a research decision. Directly compare ordered records, threshold,
prediction values, shapes, dtype, finiteness, ranges, and applicable metric
tolerances before accepting the Test package.

For PlainMLP, persist the two `float64` columns returned by `predict_proba`
without rounding, clipping, or reconstructing the denied probability. Require
shape `(n_rows, 2)`, finite values in `[0, 1]`, and row sums under `rtol = 0`,
`atol = 1.0e-7`. Validation inference and final Test inference each run once.
Freeze the Validation-selected threshold before Test. Reuse the existing locked
bootstrap indices; compare replicate IDs and valid/invalid payloads exactly,
`calibration_slope` after six-decimal rounding, other bootstrap metrics and CI
endpoints with `rtol = 0`, `atol = 1.0e-14`, and point metrics with `rtol = 0`,
`atol = 1.0e-12`. Never use SHA/digest identity for numerical equivalence.

## 12. Operator handoff checklist

For every transfer between machines, sessions, or operators, provide:

- active authorization envelope and the next permitted stage;
- repository/source location and directly recorded feature, model, runtime,
  split, and preprocessing schema/configuration fields;
- interpreter, library, CUDA, driver, GPU, device, and batch-size fields;
- `run_id`, latest validated manifest, working/final directory, and direct
  schema/content validation result;
- exact partition accessed and confirmation that later partitions remain
  closed;
- working/final directory, run state, last complete epoch or fold, and any
  retained failed-attempt directory;
- history extraction method and completeness;
- locked epoch/stopping decision with its directly referenced evidence;
- raw `1111` counts, normalized `Exempt` counts, equality-gate result, and
  semantic `1111` count;
- unresolved failure and the gate type it belongs to;
- the exact validator or next command to run.

Do not leave a decision or completion claim only in chat, terminal output, a
plot, or a notebook narrative. Persist it in the appropriate manifest or
decision artifact and validate it before handoff.

## 13. Locked decision register

This register preserves approved design decisions and output contracts. Its
statuses are decision states, not execution results. Current runtime and model
stage status is maintained only in [STATUS](../STATUS.md). An evidence link
below identifies the retained config or artifact for a decision that has
execution evidence; it does not turn this runbook into an execution-status
register.

Legacy evidence links in this register predate the final 15-feature retraining
unless a final `run_id` path explicitly records otherwise. Legacy links may
preserve decision rationale, but they are not compatible preprocessing,
fitted-model, prediction, threshold, or completion evidence for a final run.

| Decision ID | Decision status | Meaning | Retained config or execution evidence |
| --- | --- | --- | --- |
| `NEURAL-HOST-01` | `LOCKED` | Prepare locally and execute on a separate local GPU host, without a Colab dependency. | [Runtime manifest](../../../artifacts/dataset_b/realmlp_v1/runtime/runtime_manifest.json) |
| `NATIVE-INPUT-01` | `LOCKED` | Use the shared neural semantic layer with backend-owned categorical encoding. | [RealMLP input contract](../../../config/dataset_b/realmlp_v1.yaml) |
| `MODEL-SEQUENCE-01` | `LOCKED` | Implement and evaluate RealMLP before opening TabMmini. | — |
| `PILOT-SPLIT-01` | `LOCKED` | Use a stratified Train-only 20% pilot, split internally 80/20, with seed 42. | [Pilot split manifest](../../../artifacts/dataset_b/realmlp_v1/pilot/pilot_split_manifest.json) |
| `REALMLP-PILOT-START-01` | `LOCKED_FINAL_FIXED_EPOCH_50` | The complete Train-only pilot used the locked starting configuration and selected fixed epoch 50; CV and full-Train reused that epoch unchanged. | [Final pilot decision](../../../artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/pilot/decision.json) |
| `GPU-COMPAT-01` | `LOCKED_GATE` | Verify the selected GPU environment and a finite CUDA calculation before the pilot; prohibit silent CPU fallback and bind any PASS evidence to that environment. | [Runtime manifest](../../../artifacts/dataset_b/realmlp_v1/runtime/runtime_manifest.json) and [smoke result](../../../artifacts/dataset_b/realmlp_v1/runtime/smoke_result.json) |
| `RUNTIME-OBSERVABILITY-01` | `LOCKED` | Require authoritative local monitoring artifacts; W&B remains optional. | [Runtime history](../../../artifacts/dataset_b/realmlp_v1/runtime/training_history.json) |
| `VISUAL-TRAIN-01` | `LOCKED` | Produce the local post-run or post-fold training visuals defined by the runbook. | [Pilot history visual](../../../artifacts/dataset_b/realmlp_v1/pilot_run/plots/pilot_history.svg) |
| `TRAINING-HISTORY-01` | `LOCKED_OUTPUT_CONTRACT` | Use the callback/backend logger contract for complete per-epoch history; this decision does not imply that a run completed. | [Pilot training history](../../../artifacts/dataset_b/realmlp_v1/pilot_run/training_history.json) |
| `EARLY-STOPPING-01` | `LOCKED_FIXED_EPOCH_FROM_PILOT` | Select one fixed epoch from the complete Train-only pilot trajectory and reuse it unchanged for CV and full-Train; the final RealMLP evidence selected epoch 50. | [Final pilot decision](../../../artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/pilot/decision.json) |
| `CHECKPOINT-RESUME-01` | `LOCKED` | Do not resume fitted model/preprocessor state; restart a failed stage from fresh state, and use a completed fold from the same current `run_id` only as validated OOF evidence. | [CV fold 01 run state](../../../artifacts/dataset_b/realmlp_v1/cv/fold_01/run_state.json) |
| `FOLD-ATOMIC-WRITE-01` | `LOCKED` | Validate each run-scoped fold directly and publish it with a same-filesystem rename before accepting its artifacts. | [CV fold 01 manifest](../../../artifacts/dataset_b/realmlp_v1/cv/fold_01/manifest.json) |
| `REALMLP-CV-01` | `LOCKED_TRAIN_ONLY_CV` | Use fresh state for each fold in five-fold Train-only CV. | [CV summary manifest](../../../artifacts/dataset_b/realmlp_v1/cv/summary/manifest.json) and [CV results](../../../artifacts/dataset_b/realmlp_v1/cv/summary/cv_results.json) |
| `TABMMINI-OPEN-01` | `LOCKED_SATISFIED` | The RealMLP prerequisite and separate authorization were satisfied; TabMmini v1 source and final execution are closed under their own policy and evidence. | [TabMmini v1 training runbook](tabmmini_training.md) and [STATUS](../STATUS.md) |

### 13.1 PlainMLP global stop rules

Stop the current PlainMLP stage and do not open the next one when any of the
following is true:

- required repair falls outside the authorized Write Scope;
- the quality gate rejects a record, or split membership, order, or cardinality
  differs from 205,310 / 43,995 / 43,996;
- the schema differs from 15 raw / 19 semantic / 62 encoded columns;
- a PlainMLP matrix or model tensor is not `float64`, CUDA is unavailable, or
  execution silently falls back to CPU;
- probabilities are rounded, clipped, reconstructed, malformed, non-finite,
  out of range, or fail the locked row-sum tolerance;
- project Validation or Test is read before authorization, an inner-stop set
  contains outer fold-validation or project Validation/Test, or temporary
  selector state is reused by a final estimator;
- OOF coverage is not exactly once per Train row, final Test inference occurs
  more than once, or the threshold changes after Validation;
- locked bootstrap indices are regenerated or numerical comparison uses
  SHA/digest identity;
- a required validation or test fails or remains incomplete;
- the target working directory already exists and may be overwritten; or
- the final scope review or hallucination/overclaim review is not `PASS`.

## 14. Source traceability

This runbook was reconciled against:

- [Documentation migration record](../archive/documentation_migration_2026-09.md),
  which identifies `Neural_Training_Guideline.md` as a former source;
- the three handbook contracts linked in Section 1;
- the RealMLP runtime and history interfaces under
  [`src/dataset_b/runtime`](../../../src/dataset_b/runtime/);
- the RealMLP Validation, threshold, and Test utilities under
  [`src/dataset_b/evaluation`](../../../src/dataset_b/evaluation/);
- [`runtime_colab_gpu_v1.yaml`](../../../config/dataset_b/runtime_colab_gpu_v1.yaml)
  for the separately scoped Colab runtime profile.

The runbook describes procedure and interfaces. It intentionally contains no
model status, result metric, selected epoch, selected threshold, or claim that
an uninspected model family has been implemented.
