# Dataset B neural training runbook

## 1. Purpose and ownership

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
interfaces. PlainMLP, TabMmini, or a future neural model may reuse this process
only after it has its own approved configuration and equivalent runtime and
validation interfaces. RealMLP parameters must not be copied into another model
family by implication.

## 2. Authorization envelope and gate types

Before changing code or running a stage, record the authorized model, files,
partitions, decisions, execution stages, and validation activities. Apply these
gate types consistently:

| Gate type | Examples | User approval rule |
| --- | --- | --- |
| Internal verification | Static review, unit tests, manifest/hash validation, schema and dtype checks, no-training CLI checks | Does not create a new authorization gate. Run it whenever it is required inside an already authorized scope. |
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
   preprocessing, and model configuration identities.
2. Run internal no-training checks.
3. Run the GPU preflight and compatibility/history smoke.
4. Materialize or validate immutable Train-only pilot IDs.
5. Run the pilot without project Validation or Test access.
6. Review pilot evidence and lock exactly one epoch/stopping policy.
7. Run a one-fold orchestration and atomic-publication smoke when the runtime is
   new or the publication path changed.
8. Run or resume Train-only cross-validation with fresh fitted state per fold.
9. Reinitialize preprocessing and model state, then fit full Train using the
   locked policy.
10. Open Validation only under its data-boundary authorization; generate one
    ordered prediction artifact and select the threshold under the evaluation
    contract.
11. Open Test only under its separate data-boundary authorization and only
    after the model, preprocessing, and threshold are frozen.

Pilot and smoke outputs are not CV, Validation, Test, or RQ1 evidence. Outer
fold-validation may be predicted once after a fold fit, but it must not control
the epoch budget, stopping rule, preprocessing, or backend-fitted state.

## 4. Pre-run checks

From the repository root:

1. Confirm the active configuration and source hashes that will be recorded in
   manifests.
2. Confirm the immutable `split_v1` membership and order; do not regenerate it
   from a model runner.
3. Confirm that the model consumes the neural representation assigned by the
   preprocessing contract.
4. Run the relevant unit and no-training tests. A test that imports only a CPU
   build is not GPU evidence.
5. Ensure the artifact destination is on the filesystem on which the final
   same-filesystem rename will occur.
6. Check free disk space for the temporary directory, model, histories,
   predictions, hashes, and retained failed attempts.
7. Confirm that no final output directory will be overwritten.

Record failures in the handoff. Do not repair a feature, config, or policy
difference as incidental cleanup; route it through the research-decision gate.

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

## 6. Pilot procedure

Pilot data must come only from immutable Train membership. The pilot-fit and
pilot-inner-stop ID files must be disjoint, have exact membership and hashes,
and contain no project Validation or Test IDs. Use the public split functions
in
[`realmlp_pilot.py`](../../../src/dataset_b/splits/realmlp_pilot.py) to
materialize once or validate an existing pilot split; do not regenerate the IDs
per attempt.

Run the RealMLP pilot through its public CLI:

```console
python -m src.dataset_b.runtime.run_realmlp_pilot run
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
history hash, and the same full-Train refit policy. CV folds and full-Train fit
must use the locked budget consistently, must not create an inner-stop set, and
must represent inner-validation and best-epoch history fields as not
applicable.

### Train-only inner stopping

Choose `INNER_EARLY_STOPPING` only when the backend-supported procedure is
fully specified. Record the inner split fraction and seed, patience parameters,
metric, maximum budget, refit behavior, and evidence hash. Each fold must derive
the stopping set from fold-train only. Full-Train refit must derive any stopping
set from Train only. Project Validation is never an early-stopping channel.

If the curve remains unstable or the optimum is unresolved at the budget
boundary, do not guess. Propose another Train-only pilot budget or approved seed
as a research-decision change.

The `decide` command mutates the policy and publishes the pilot, so run it only
after the chosen policy is authorized or when that exact choice is already
inside the active authorization:

```console
python -m src.dataset_b.runtime.run_realmlp_pilot decide --decision FIXED_EPOCH --fixed-epochs <N> --rationale "<evidence-based rationale>"
python -m src.dataset_b.runtime.run_realmlp_pilot decide --decision INNER_EARLY_STOPPING --inner-stop-fraction <F> --inner-stop-seed <SEED> --additive-patience <N> --multiplicative-patience <X> --rationale "<evidence-based rationale>"
python -m src.dataset_b.runtime.run_realmlp_pilot validate
```

Use one decision command, not both.

The current RealMLP CV smoke, production CV, and full-Train runners implement
the fixed-epoch branch. If a pilot selects `INNER_EARLY_STOPPING`, stop after
publishing and validating that decision. Do not use the fixed-epoch runners
until a separately authorized implementation has added the approved inner
split/stopping behavior and matching validators.

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

## 9. CV and full-Train execution

After the pilot decision and runtime gate validate, the current RealMLP
interfaces are:

```console
python -m src.dataset_b.runtime.run_realmlp_cv_smoke run
python -m src.dataset_b.runtime.run_realmlp_cv_smoke validate
python -m src.dataset_b.runtime.run_realmlp_cv run
python -m src.dataset_b.runtime.run_realmlp_cv validate
python -m src.dataset_b.runtime.run_realmlp_full_train run
python -m src.dataset_b.runtime.run_realmlp_full_train validate
```

Each CV fold must initialize fresh preprocessing, model, optimizer, and
backend-fitted state. Publish ordered fold predictions once after training;
aggregate OOF evidence only after every required fold validates. Full-Train
must also start from fresh state and consume only Train membership. A completed
CV model must not be reused as the full-Train model.

## 10. Atomic publication and crash recovery

Write a run or fold to a sibling temporary directory on the same filesystem as
its final destination. During execution, the temporary directory is diagnostic
working state only and must not feed OOF aggregation or downstream evaluation.

Before publication:

1. finish training and the stage-appropriate prediction;
2. verify row count, exact ordered `record_id` membership, probability shape,
   finiteness, and range;
3. verify model, history, predictions, result, and manifest contents;
4. hash every required payload;
5. write `COMPLETED` last and close open file handles;
6. validate the complete temporary bundle;
7. rename the whole directory to its final name on the same filesystem;
8. validate the final directory again.

The marker inside a temporary directory does not make it published. A model
file alone is never completion evidence.

On restart:

- validate and skip a final directory only when its marker, manifest, hashes,
  IDs, history, and probabilities all pass;
- preserve an interrupted temporary directory as a numbered failed attempt;
- start a clean temporary directory instead of mixing attempts;
- resume at the completed run/fold boundary unless the model-specific contract
  explicitly supports and verifies finer-grained checkpoint recovery;
- never overwrite a valid published directory.

## 11. Validation and Test handoff

The following commands exist for the current RealMLP evaluation path, but they
must be run only after the corresponding data-boundary authorization:

```console
python -m src.dataset_b.evaluation.realmlp_validation run
python -m src.dataset_b.evaluation.realmlp_validation validate
python -m src.dataset_b.evaluation.realmlp_validation_threshold run
python -m src.dataset_b.evaluation.realmlp_validation_threshold validate
python -m src.dataset_b.evaluation.realmlp_test_evaluation run
python -m src.dataset_b.evaluation.realmlp_test_evaluation validate
```

Before opening Validation, hand off the validated full-Train manifest and
fitted-model hash, split/config/preprocessing identities, ordered Validation
membership, probability convention, and confirmation that fitting is complete.
Threshold selection may inspect Validation labels only for the approved
objective and may not refit preprocessing or model state.

Before opening Test, hand off the validated Validation prediction and threshold
artifacts and hashes, frozen model/preprocessing/configuration, ordered Test
membership, and bootstrap identity. Test execution must not fit, change the
threshold, regenerate shared bootstrap membership, or reopen a research
decision.

## 12. Operator handoff checklist

For every transfer between machines, sessions, or operators, provide:

- active authorization envelope and the next permitted stage;
- repository/source identity and hashes of feature, split, preprocessing,
  model, and runtime configs;
- interpreter, library, CUDA, driver, GPU, device, and batch-size identity;
- latest validated manifest and artifact hashes;
- exact partition accessed and confirmation that later partitions remain
  closed;
- working/final directory, run state, last complete epoch or fold, and any
  retained failed-attempt directory;
- history extraction method and completeness;
- locked epoch/stopping decision with its evidence hash;
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

| Decision ID | Decision status | Meaning | Retained config or execution evidence |
| --- | --- | --- | --- |
| `NEURAL-HOST-01` | `LOCKED` | Prepare locally and execute on a separate local GPU host, without a Colab dependency. | [Runtime manifest](../../../artifacts/dataset_b/realmlp_v1/runtime/runtime_manifest.json) |
| `NATIVE-INPUT-01` | `LOCKED` | Use the shared neural semantic layer with backend-owned categorical encoding. | [RealMLP input contract](../../../config/dataset_b/realmlp_v1.yaml) |
| `MODEL-SEQUENCE-01` | `LOCKED` | Implement and evaluate RealMLP before opening TabMmini. | — |
| `PILOT-SPLIT-01` | `LOCKED` | Use a stratified Train-only 20% pilot, split internally 80/20, with seed 42. | [Pilot split manifest](../../../artifacts/dataset_b/realmlp_v1/pilot/pilot_split_manifest.json) |
| `REALMLP-PILOT-START-01` | `PROVISIONAL` | Freeze the documented starting configuration for the pilot only. | [RealMLP configuration](../../../config/dataset_b/realmlp_v1.yaml) |
| `GPU-COMPAT-01` | `LOCKED_GATE` | Verify the selected GPU environment and a finite CUDA calculation before the pilot; prohibit silent CPU fallback and bind any PASS evidence to that environment. | [Runtime manifest](../../../artifacts/dataset_b/realmlp_v1/runtime/runtime_manifest.json) and [smoke result](../../../artifacts/dataset_b/realmlp_v1/runtime/smoke_result.json) |
| `RUNTIME-OBSERVABILITY-01` | `LOCKED` | Require authoritative local monitoring artifacts; W&B remains optional. | [Runtime history](../../../artifacts/dataset_b/realmlp_v1/runtime/training_history.json) |
| `VISUAL-TRAIN-01` | `LOCKED` | Produce the local post-run or post-fold training visuals defined by the runbook. | [Pilot history visual](../../../artifacts/dataset_b/realmlp_v1/pilot_run/plots/pilot_history.svg) |
| `TRAINING-HISTORY-01` | `LOCKED_OUTPUT_CONTRACT` | Use the callback/backend logger contract for complete per-epoch history; this decision does not imply that a run completed. | [Pilot training history](../../../artifacts/dataset_b/realmlp_v1/pilot_run/training_history.json) |
| `EARLY-STOPPING-01` | `LOCKED_FIXED_EPOCH_51` | Use 51 fixed epochs and no inner-stop set for CV and full-Train. | [Pilot decision](../../../artifacts/dataset_b/realmlp_v1/pilot_run/decision.json) |
| `CHECKPOINT-RESUME-01` | `LOCKED` | Resume only at a completed run or fold boundary. | [CV fold 01 run state](../../../artifacts/dataset_b/realmlp_v1/cv/fold_01/run_state.json) |
| `FOLD-ATOMIC-WRITE-01` | `LOCKED` | Validate each fold and publish it with a same-filesystem rename before accepting its artifacts. | [CV fold 01 manifest](../../../artifacts/dataset_b/realmlp_v1/cv/fold_01/manifest.json) |
| `REALMLP-CV-01` | `LOCKED_TRAIN_ONLY_CV` | Use fresh state for each fold in five-fold Train-only CV. | [CV summary manifest](../../../artifacts/dataset_b/realmlp_v1/cv/summary/manifest.json) and [CV results](../../../artifacts/dataset_b/realmlp_v1/cv/summary/cv_results.json) |
| `TABMMINI-OPEN-01` | `LOCKED_GATE` | Require the RealMLP prerequisite review, separate authorization, and a model-specific configuration before opening TabMmini. | — |

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
