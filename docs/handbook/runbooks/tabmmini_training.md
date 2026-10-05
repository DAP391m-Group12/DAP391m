# Dataset B TabMmini v1 training runbook

## 1. Purpose and status boundary

This runbook is the authoritative model-specific decision lock for the final
Dataset B TabMmini v1 implementation. It fixes the architecture, preprocessing
tail, objective, ensemble semantics, pilot, runtime, recovery, prediction, and
evaluation handoff before source implementation begins.

These are approved design and implementation requirements, not execution
evidence. Until [STATUS](../STATUS.md) records evidence from the corresponding
stage, this document does not establish that the configuration, adapter,
backend, CUDA runtime, pilot, training, inference, or artifacts exist or pass.

TabMmini is a new model family. It must not reuse or warm-start PlainMLP or
RealMLP model weights, optimizer state, checkpoints, fitted preprocessing,
category state, predictions, thresholds, OOF outputs, or run identifiers.

The shared boundaries remain owned by:

- [Neural preprocessing](../contracts/preprocessing_neural.md) for raw 15,
  normalized 15, semantic 19, fit scope, category support, and adapter output;
- [Split and evaluation](../contracts/split_and_evaluation.md) for `split_v1`,
  five-fold Train-only CV, Validation thresholding, Test evaluation, metrics,
  and bootstrap;
- [Neural training](neural_training.md) for common runtime, publication,
  recovery, observability, and handoff rules.

## 2. Locked backend and model configuration

Use the stable official package and public constructor:

```yaml
backend:
  package: tabm
  package_version: "0.0.3"
  constructor: tabm.TabM.make

model:
  arch_type: tabm-mini
  k: 32
  n_blocks: 2
  d_block: 512
  dropout: 0.1
  d_out: 1
  num_embeddings: null

optimizer:
  name: AdamW
  lr: 0.002
  weight_decay: 0.0003

training:
  class_weight: true
  requested_batch_size: 256
  batch_fallback: [128, 64]
  seed: 42
  stopping_policy: fixed_epoch_after_pilot
  early_stopping: false

evaluation:
  cv: StratifiedKFold_5_train_only
  fresh_model_each_fold: true
  full_train_fresh_model: true
  hyperparameter_search: false
```

Do not run GridSearch, Optuna, or another configuration trial. A CUDA OOM may
change only the effective batch size under Section 7; it must not change `k`,
width, depth, dropout, learning rate, optimizer, or architecture.

The implementation uses `float32` model computation. AMP and `torch.compile`
are disabled for v1. The runtime records the installed Python, TabM, PyTorch,
CUDA, driver, GPU, and device versions rather than assuming compatibility from
the declared package version.

## 3. Locked preprocessing and categorical boundary

The route is:

```text
raw 15
-> central quality gate and 1111 -> Exempt normalization
-> normalized 15
-> fit-local semantic 19
-> log1p on income_value and loan_amount_value only
-> TabMmini native adapter
-> TabM backend
```

Median state is fitted independently on pilot-fit, each current fold-train, or
full Train, according to the active stage. No StandardScaler, RobustScaler, or
project/external OHE is allowed. Numeric tensors are finite `float32` and
categorical tensors are integer/`long`.

The adapter maps each authoritative semantic schema category to a fixed integer
index. It does not learn a vocabulary from pilot data, full Train, or a fold.
The same map, including preserved schema categories absent from a particular
fit population, is used for the pilot, all five folds, and full Train. `Exempt`
is retained where allowed; `1111`, `MISSING`, and unknown categories are
forbidden. There is no `<UNK>` category or silent remapping.

Integer indexing is the native input representation, not project-level OHE.
The indexed categorical tensors are passed once to the TabM package, whose
native categorical encoding is backend-owned. This prevents double encoding:

```yaml
categorical:
  vocabulary_source: authoritative_schema
  mapping: fixed_schema_category_to_index
  learn_vocabulary_from_data: false
  adapter_output: fixed_schema_integer_indices
  external_ohe: false
  backend_encoding: tabm_native
  allow_backend_internal_ohe: true
  double_encoding: false
  unknown_category: fail
  preserve_absent_schema_categories: true
  support_audit: true
```

Each fold still records a support audit:

| Fold-train support | Fold-validation support | Outcome |
| --- | --- | --- |
| Present | Any | `PASS`; record counts |
| Absent | Absent | `WARN`; preserve schema position |
| Absent | Present | `FAIL` before training |
| Outside authoritative schema | Any | `FAIL` |

## 4. Locked target, loss, and ensemble semantics

The target mapping is fixed:

```text
y_approved = 1 -> Approved
y_approved = 0 -> Denied
```

The objective contract is:

```yaml
loss:
  name: BCEWithLogitsLoss
  positive_label: 1
  positive_class: approved
  pos_weight:
    formula: n_denied / n_approved
    pilot_population: pilot_fit_only
    cv_population: current_fold_train_only
    full_train_population: full_train_only
  sample_weight: false
  resampling: false
  member_loss:
    reduction_before_final: none
    final_reduction: mean_all_elements
    average_logits_before_loss: false

target:
  input_shape: [B]
  member_shape: [B, 32, 1]
  dtype: float32

ensemble:
  k: 32
  training:
    target_broadcast: expand_to_logits
    loss: mean_member_losses
  inference:
    member_transform: sigmoid
    aggregation: mean_probability
    average_logits_before_sigmoid: false
```

The model output, expanded target, unreduced loss, and scalar loss must have
shapes `[B, 32, 1]`, `[B, 32, 1]`, `[B, 32, 1]`, and `[]`, respectively. The
canonical training operation is equivalent to:

```python
logits = model(x_num, x_cat)
targets = y.float().view(-1, 1, 1).expand_as(logits)
pos_weight = torch.tensor(
    [n_denied / n_approved],
    dtype=logits.dtype,
    device=logits.device,
)
element_losses = torch.nn.functional.binary_cross_entropy_with_logits(
    logits,
    targets,
    pos_weight=pos_weight,
    reduction="none",
)
loss = element_losses.mean()
```

The implementation must reject a fit population that does not contain both
classes. It must record the population, `n_approved`, `n_denied`, formula, and
resulting `pos_weight` in the stage manifest. Pilot-stop, outer
fold-validation, project Validation, and Test never contribute to this weight.

Loss on mean logits is forbidden. The 32 member losses are formed first and
then reduced equally over samples, members, and the single output dimension.

## 5. Locked Train-only pilot and epoch decision

Run exactly one pilot, entirely within immutable Train membership:

```yaml
pilot:
  source: train_only
  sample_fraction: 0.20
  stratified: true
  seed: 42
  split:
    fit_fraction: 0.80
    stop_fraction: 0.20
    stratified: true
    seed: 42
  max_epochs: 100
  selection_metric: inner_stop_weighted_bce_loss
  selection_mode: min
  metric_pos_weight_source: pilot_fit_only
  metric_member_reduction: mean_all_elements
  epoch_numbering: one_based
  tie_tolerance:
    rtol: 0.0
    atol: 1.0e-6
  tie_break: earliest_epoch
  boundary_rule: unresolved_if_best_epoch_is_100_and_curve_is_still_improving
```

Preprocessing, model state, optimizer state, and `pos_weight` are fitted only
from pilot-fit. Each epoch evaluates pilot-stop by forward-only weighted
member-wise BCE using the same pilot-fit `pos_weight`. Pilot-stop does not fit
or update preprocessing, category maps, model parameters, optimizer state, or
class weight.

The selected fixed epoch `E` is the one-based epoch with minimum
`inner_stop_weighted_bce_loss`; losses are tied only when they differ by at
most `1.0e-6` under `rtol = 0`, and a tie chooses the earliest epoch. Training loss,
ROC-AUC, outer fold-validation, project Validation, and Test cannot select
`E`. If epoch 100 is best and the curve is still improving at the boundary,
the decision is `UNRESOLVED`; do not guess `E` or open CV without a newly
authorized decision.

After a valid decision, freeze all model and training settings. Each of the
five CV folds and the fresh full-Train model trains for exactly `E` epochs with
early stopping disabled and no inner-stop channel.

## 6. Fit scope and fresh-state requirements

Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on Train only.
Every fold creates fresh preprocessing, model, optimizer, DataLoader, and
backend-owned state. Recompute median state and
`pos_weight = n_denied / n_approved` from that fold's fold-train rows only.
Fold-validation is transformed and predicted once but never updates state.

Full Train creates another fresh preprocessing pipeline, model, and optimizer.
It recomputes medians and `pos_weight` from all Train rows and fits exactly
`E` epochs. Pilot or fold state must not initialize full Train. Validation and
Test must not be passed as training, stopping, weighting, or vocabulary input.

Training DataLoaders shuffle deterministically; evaluation DataLoaders do not
shuffle. `drop_last` is false. Seed Python, NumPy, PyTorch, and workers from
the locked seed `42`, and record observed deterministic-runtime settings.

## 7. CUDA gate and whole-run batch lock

Execution is CUDA-only on device index `0`; silent CPU fallback is forbidden.
Before the pilot, run a preflight and two-epoch smoke that verifies package
imports, a finite CUDA operation, model output `[B, 32, 1]`, finite forward and
member-wise weighted loss, backward and AdamW step, inference aggregation, and
serialization reload.

The smoke tries batch sizes only in this order:

```text
256 -> 128 -> 64
```

Fallback is permitted only after a confirmed CUDA OOM. The first passing size
becomes `effective_batch_size` for the whole run: pilot, all five CV folds, and
full Train. Folds may not choose sizes independently. If this locked size later
OOMs in production, fail the stage; do not silently reduce it or change another
parameter. CUDA availability, the effective batch size, and the selected epoch
`E` are runtime results, not unlocked architecture decisions.

## 8. Prediction and evaluation handoff

Canonical inference applies sigmoid to each member before aggregation:

```python
logits = model(x_num, x_cat)                  # [B, 32, 1]
member_probs = torch.sigmoid(logits).squeeze(-1)  # [B, 32]
p_approved = member_probs.mean(dim=1)         # [B]
```

Applying sigmoid after averaging logits is forbidden. The canonical persisted
prediction is:

```yaml
probability:
  name: p_approved
  meaning: P(y_approved=1)
  positive_label: 1
  dtype: float32
  shape: [n]
```

When a shared evaluator requires two columns, the evaluation adapter may
derive `[1.0 - p_approved, p_approved]`, where column 0 is `P(Denied)` and
column 1 is `P(Approved)`. This derived matrix does not replace the canonical
single-vector TabMmini prediction.

Validation and Test inference each occur once for a `run_id`. Validation uses
the shared threshold grid, objective, and tie rule in the split/evaluation
contract, freezes the selected threshold, and performs no refit. Test uses the
frozen model and threshold once. Evaluation reuses the existing 1,000 locked
bootstrap indices and shared numerical-comparison policy; it does not
regenerate indices or retrain inside a replicate.

## 9. Artifacts, recovery, and serialization

One unique `run_id` links preprocessing, runtime smoke, pilot, CV, full Train,
Validation, Test, reporting, and review. Write under a run-scoped working
directory and publish atomically to the final directory only after direct
validation. Do not overwrite a final artifact.

Do not resume model or optimizer state after an interrupted stage. Retain the
failed attempt for diagnosis and restart that stage with fresh preprocessing,
model, and optimizer state. A completed and validated fold from the same run
may retain its prediction for OOF aggregation, but its weights and fitted state
must not initialize another fold or full Train.

The final inference bundle contains at least:

- model `state_dict`;
- complete TabMmini architecture and training configuration;
- fixed category-to-index maps and ordered semantic schema;
- fitted preprocessing state for full Train;
- effective batch size and fixed epoch `E`;
- actual library and runtime versions;
- `run_id` and directly reviewable manifest.

Optimizer state is not part of the final inference bundle. Save/load validation
must verify configuration, schema, dtypes, category maps, and finite prediction
equivalence under an explicitly recorded numerical tolerance.

## 10. Stop rules

Stop the current stage without changing the locked policy when:

- the 15-to-19 semantics, selective `log1p`, fixed category maps, or
  normalization differ from the contracts;
- a category is unknown or appears in fold-validation while absent from the
  corresponding fold-train population;
- CUDA is unavailable, CPU fallback occurs, or all permitted batch sizes OOM;
- output, target, loss, or probability shapes differ from this runbook;
- logits, losses, gradients, probabilities, or serialized predictions are
  non-finite;
- a fit population lacks either class;
- the pilot decision is unresolved at epoch 100;
- outer fold-validation, project Validation, or Test affects model fitting,
  preprocessing, class weight, epoch selection, or configuration;
- the locked effective batch size, fixed epoch, architecture, optimizer, or
  ensemble semantics change;
- Validation or Test inference would be repeated for the same run;
- locked bootstrap indices would be regenerated; or
- publication would overwrite an existing final artifact.

## 11. Locked decision register

| Decision ID | Status | Locked meaning |
| --- | --- | --- |
| `TABMMINI-BACKEND-01` | `LOCKED` | Use `tabm==0.0.3` through `tabm.TabM.make`. |
| `TABMMINI-ARCH-01` | `LOCKED` | Use `tabm-mini`, `k=32`, two 512-wide blocks, dropout 0.1, one output, and no numerical embeddings. |
| `TABMMINI-NATIVE-01` | `LOCKED` | Consume native semantic 19 with numeric `float32`, fixed-schema categorical indices, no external scaler/OHE, and one backend-owned categorical encoding. |
| `TABMMINI-LOSS-01` | `LOCKED` | Use fit-population weighted member-wise BCE and never average logits before loss. |
| `TABMMINI-PILOT-01` | `LOCKED` | One 20%-Train pilot with an 80/20 inner split selects one fixed `E` from weighted inner-stop BCE, subject to the epoch-100 boundary rule. |
| `TABMMINI-BATCH-01` | `LOCKED_GATE` | CUDA smoke chooses one whole-run batch size through `256 -> 128 -> 64` only on confirmed OOM. |
| `TABMMINI-INFERENCE-01` | `LOCKED` | Apply sigmoid per member and average probabilities into canonical `p_approved`. |
| `TABMMINI-FRESH-STATE-01` | `LOCKED` | Pilot, every fold, and full Train use their required fresh state; no model or optimizer resume. |
| `TABMMINI-EVAL-01` | `LOCKED` | Use one-pass holdout inference, the shared threshold policy, and existing locked bootstrap indices. |
