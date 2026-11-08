# Dataset B — Current Execution Status

**Reconciled:** 2026-11-09
**Authority:** This file is the sole register of current execution status. Other project documents define contracts, procedures, or history; status claims found elsewhere are not authoritative unless they are clearly labeled as historical and point back here.

## Current Dataset B reproduction route — 2026-11-03

The version 2.3 reproduction implementation is present at source commit
`4bdfa0a5aebe138982b75a055891c1321a4d5496` on `main`. Claude Code implemented
the route; the commit was authored and committed by Dan with Claude Opus 5.5
recorded as co-author. This attribution records historical implementation work and
does not assign Claude a current implementer or verifier role.

| Component | Status | Evidence and limitation |
|---|---|---|
| Implementation | `COMPLETE` | Commit `4bdfa0a5aebe138982b75a055891c1321a4d5496` contains `src/dataset_b/reproduce/`, the asset manifest, five environment profiles, wrapper, guide, and focused tests. No model training or fitting is performed by the reproduction route. |
| Documentation and focused static/unit validation | `PASS` | The version 2.3 guide and acceptance register pin the source commit, resolve their internal links, preserve the 0–12 rubric, and are covered by the focused documentation suite. This is static/unit evidence, not an end-to-end runtime result. |
| Historical author-machine observations | `PARTIAL` | On 2026-11-03, `environment`, `verify-assets`, `evaluate`, `compare`, and `reports` were observed from a working tree using verified local bundles. Classical and neural evaluation ran in separate invocations after an initial defect was fixed. `reports` exited 1 with `INCOMPLETE`. |
| Current-host technical resolution | `COMPLETE` | The 2026-11-05 issue was identified and recorded: this machine has no CUDA and does not contain the locked virtual environments; the owner records that those environments are on another machine. The asset gate passed 654/654, no checkpoint or metric comparison failed, and Phase B/C were stopped by owner decision. This status applies only to resolution of this host/environment issue. |
| CUDA handoff Phase B, 2026-11-08 to 2026-11-09 | `COMPLETE` | At commit `5a5667b13546e6b088756cd8e9da8e963a0ccfe0` plus the scoped TabMmini reproduction fix on `codex/cuda-phase-b-c-handoff`, assets, all five environments, six-model frozen-checkpoint evaluation, all 402 RQ1 rows, and RQ2/RQ3/RQ4a/RQ4b passed. RQ2 passed 28/28 after its CPU inference thread count was locked to one and restored after each call. `reports` returned the allowed documented `INCOMPLETE` result: 5 objects PASS and 3 `NOT_REGENERATED`. Phase C was not opened by this task. |
| CUDA handoff Phase C build and LFS publication, 2026-11-09 | `COMPLETE` | `maintainer-build --source-root .` rebuilt 11 core ZIPs totaling 678,068,350 bytes from 654 assets. All size, SHA-256, and member-inventory checks passed against the prebuild manifest; RQ3 `runtime.csv` and the optional RQ4 audit archive are absent from core. Release commit `1d384482f7b00f84c06cdecbb4abd13380da91a2` and all 11 core LFS objects were pushed to `codex/cuda-phase-b-c-handoff`; post-push `git lfs fsck` passed. Public URLs, anonymous download, and clean-clone verification remain separate gates. |
| Overall reproduction runtime | `PARTIAL` | The current-source end-to-end clean-clone reproducibility is not established. Closing the current-host Phase B/C attempt does not mark the entire reproduction runtime complete. |

### CUDA handoff Phase B — 2026-11-08 to 2026-11-09

The attempt ran from commit `5a5667b13546e6b088756cd8e9da8e963a0ccfe0`
on `codex/cuda-phase-b-c-handoff`. Static inspection of every reachable Phase B
entrypoint preceded execution. Imported modules that also contain training functions
were traced to the inference-only helpers actually called. No training, fitting,
refitting, threshold selection, lambda selection, or seed selection route ran.

`verify-assets` exited 0 and passed 654/654 assets with no missing or mismatched
entries. After the owner authorized installation, `environment --install` created
the five locked profiles and the environment gate passed all five. The three CUDA
profiles used Torch `2.14.0+cu130`, CUDA build `13.0`, cuDNN `92400`, driver
`610.74`, and each completed a finite CUDA tensor operation on the NVIDIA GeForce
RTX 5050 Laptop GPU with 8,546,484,224 bytes of device memory. The retained receipts
are `verify-assets_20261108T122017_953903Z.json` and
`environment_20261108T125329_497165Z.json` under
`reports/reproducibility/receipts/`.

`evaluate --model all` then passed six of six frozen-checkpoint evaluations. Every
model receipt records zero model-fit calls, zero preprocessing-fit calls, one
checkpoint-inference pass, and `training_code_called=false`. The three neural models
recorded actual `cuda:0` allocations of 35,195,392, 52,161,536, and 86,641,664 bytes.
`compare` passed all 402 rows with zero failures. The command receipts are
`evaluate_20261108T125910_329422Z.json` and
`compare_20261108T125939_503172Z.json`.

The first research run completed all four workers. RQ3 passed 5/5 comparison rows, RQ4a
passed 4/4 package-validation rows without fitting `ThresholdOptimizer`, and RQ4b
passed 17/17 rows from saved predictions without reopening training, CV, lambda, or
threshold selection. RQ2 completed all six frozen CPU-inference models and all 1,000
bootstrap replicates per model, but its final comparison failed 3 of 28 rows. All
three failures are TabMmini numerical differences above the locked `1e-12` tolerance:
`importance.csv` delta `2.12172445194711e-10`, `cube.npy` maximum delta
`5.960464477539063e-08`, and `gsv_local.npy` maximum delta
`1.9868215961338365e-09`. The research receipt is
`research_20261108T141813_840212Z.json`.

Diagnosis found 432 of 1,048,576 TabMmini cube values different, with the
maximum difference equal to one float32-scale step at the observed value. The derived
`gsv_local.npy` differs in 63 of 3,072 values, and 23 to 24 importance rows differ by
column. This pattern is consistent with environment-sensitive CPU inference rounding,
but the accepted RQ2 package does not record enough runtime provenance to confirm a
specific hardware or kernel cause from historical provenance alone. A controlled
frozen-inference check then isolated the current-host divergence: 14 Torch intra-op
threads reproduced the failed output exactly, while one thread matched the accepted
TabMmini cube block exactly; repeated calls on the same host were stable. The minimal
fix locks TabMmini CPU inference to one intra-op thread and restores the caller's
process-wide setting in a `finally` block. Its focused suite passed 51 tests.

The official `research --route all` rerun passed RQ2 28/28, RQ3 5/5, RQ4a 4/4,
and RQ4b 17/17. Its receipt is
`research_20261108T174830_904582Z.json`. A log scan found no fit, train, refit,
backward, optimizer-step, or selection event. `reports` then returned the documented
`INCOMPLETE` status with five objects PASS and three `NOT_REGENERATED`: Figures 2 and
3 and the assembled RQ text reports have no source-backed builder and their notebooks
were not executed. The reports receipt is `reports_20261108T174846_986881Z.json`.

The locked numerical policy was not changed and accepted artifacts were not modified.
Phase B is complete under the handoff rule that permits the already documented
object-level report gaps after all scientific gates pass. Phase C was not opened by
this task, and no release build, LFS staging, commit, or push was attempted.

### CUDA handoff Phase C build and LFS publication — 2026-11-09

After the Phase B scientific gates passed, static review confirmed that
`maintainer-build` only inventories retained files, hashes them, creates deterministic
ZIPs, and writes the manifest. It has no training, fitting, inference, selection, or
frozen-artifact write path. The focused asset/release/documentation suite passed 112
tests when run with a Git ceiling that kept test fixtures outside repository discovery.

The build completed with 654 assets, 635 required assets, and 11 core bundles totaling
678,068,350 bytes. The post-build comparison found the bundle identities, asset
inventory, and inventory digest unchanged from the prebuild manifest. Each ZIP passed
its declared byte-size and SHA-256 check, and its exact member list matched the assets
assigned to that bundle. RQ3 `runtime.csv` is excluded and the optional RQ4 audit
archive is not present in the core manifest. The build updated only
`source_provenance.source_revision` from the earlier implementation revision to
`5a5667b13546e6b088756cd8e9da8e963a0ccfe0`.

The verification record is
`reports/reproducibility/phase_c/phase_c_bundle_verification_20261108T180034Z.json`.
Release commit `1d384482f7b00f84c06cdecbb4abd13380da91a2` was pushed to
`origin/codex/cuda-phase-b-c-handoff`. The remote branch resolved to the same SHA,
Git LFS reported no pending objects, all 11 core objects were listed with their locked
SHA-256 identities, and post-push `git lfs fsck` passed. Phase C build/stage/commit/push
status is `COMPLETE`. Public bundle URLs and direct URLs for the three locked split
CSVs are still absent; anonymous download and clean-clone installation have not run.

### Current-host Phase B attempt — 2026-11-05

The reconciled asset gate passed with 654/654 assets verified. The next environment
gate failed before any model worker or inference process started: none of the five
locked interpreters exists under `.venv-repro/`. The host has Python 3.13.0, `uv` is
not installed, and `nvidia-smi` is unavailable. The retained `.venv-realmlp-gpu` and
`.venv-rq4a-cpu` directories are not portable environments; their `pyvenv.cfg` files
point to a missing `C:\Users\Dan\...\Python313\python.exe` base interpreter.

Evidence is retained in
`reports/reproducibility/receipts/environment_20261105T105959_367893Z.json` and
`reports/reproducibility/receipts/evaluate_20261105T105933_714932Z.json`. The evaluate
receipt lists all six models as failed on the missing-environment precondition; it is
not a checkpoint, metric, or scientific-result failure. The owner stopped and closed
all remaining Phase B and Phase C work on 2026-11-05. `compare`, RQ2, and `reports`
were not run, and release Phase C was not opened. The current machine is not the CUDA
execution host, and the owner records that the locked virtual environments are on
another machine. Resolution and documentation of this specific technical issue are
`COMPLETE`. This does not mark the full reproduction runtime complete.

### Optional RQ4 audit-evidence handoff — 2026-11-05

`scratch/` is ignored local development/review state and is not a dependency of the
core reproduction route or the frozen RQ4 production packages. A curated 243-file,
4,424,422-byte historical subset was copied into the local optional archive
`dist/reproducibility/evidence/dataset_b_rq4_audit_evidence_20261105.zip` without
deleting, moving, renaming, or overwriting its source files. The tracked
`config/dataset_b/reproducibility/rq4_audit_evidence_manifest.json` records every
member and the archive SHA-256. Pytest basetemp/fixture trees and scratch roots without
saved review logs were excluded.

This archive is audit/provenance evidence only. It is intentionally absent from the
core asset manifest and does not replace production receipts, frozen artifacts, or a
runtime reproduction result. Its public URL remains unrecorded; external publication
is a later owner delivery action.

### RQ3 volatile-runtime release reconciliation — 2026-11-05

The canonical RQ3 `runtime.csv` remains unchanged in the accepted production run and
continues to satisfy the RQ3 schema and semantic validator. It records wall-clock
timings and is not compared by the RQ3 reproduction route. A prior release inventory
had classified a different byte version as a required `rq_input`; the exact alternate
copy and its maintainer source root were not retained.

The core release policy now excludes this volatile timing file while preserving the
historical artifact. The RQ3 bundle was rebuilt from its five deterministic members,
and the manifest records source revision `4bdfa0a5aebe138982b75a055891c1321a4d5496`,
an included-inventory digest, and the excluded path. Local `verify-assets` passed
654/654 after reconciliation. This is local asset-integrity evidence only; it does not
establish public download, current-source RQ execution, or clean-clone reproduction.

The final tag, owner-supplied bundle links, anonymous download verification, and a
qualified clean-host run remain delivery gates. Figures, text reports, and notebooks
were not regenerated; Figures 4 to 6 are absent from the repository and Figure 1 has
no regeneration route. These limitations do not change the accepted frozen
experimental runs recorded elsewhere in this register.

## Current RQ4b handover ownership

### RQ4b-3 closure — 2026-10-24

RQ4b-3 (primary adversarial RealMLP analysis on G3) is **closed** for
execution, validation and reporting. The closure was reconciled directly
against the production receipts, the package report and Notebook 12. Each
status below is separate. None of them is a research conclusion.

| Component | Status | Evidence (details in the dated sections below) |
|---|---|---|
| Development execution (nine sparse units) | `COMPLETE`; Gate B `PASS` | Immutable source runs `rq4b3_adv_20261019_02` (seed 42), `rq4b3_adv_20261021_01` (seed 43) and `rq4b3_adv_20261022_01` (seed 44): 45 CV fits, nine full-Train refits, 18 `PASS` receipts |
| Gate C aggregation and Validation selection | `COMPLETE`; receipt `PASS` | `rq4b3_gatec_20261024_01` (manifest `5d072d7f...931b`, receipt `14f602bb...1448f13`, selection `76af4d07...c8c615`). The stored selection is `selected_lambda = 0.0` under the approved `candidate_selection` policy. On Validation, mitigation effectiveness is recorded as not established. |
| Production Test evaluation | `COMPLETE`; receipt `PASS` | `rq4b3_test_20261024_01` stage `test_evaluation` (receipt `2f7c1ae8...32c4`). Nine arms, nine inferences, 0 fit/refit/CV, 43,996 rows per prediction, 1,000/1,000 valid replicates for all six comparisons and three quantities |
| Production package validation | `PASS` | Stage `package_validation` (receipt `d20fa9b1...800a`, report `02af1bc6...db32403`). No inference, recomputation from saved predictions, byte- and tolerance-identical. Local fits 0; source accounting 45 CV / nine refits |
| Reporting | `COMPLETE` | `notebooks/dataset_b/final/research_questions/12_rq4b_adversarial_results.ipynb` (SHA-256 `2876b1ae...2032`): 15 sequential cells, no saved error, key values equal the stored artifacts |
| Research interpretation of Test results | Not recorded; user-owned | Code assigns no pass/fail, effectiveness, support or recommendation label. Notebook 12 section 14 is reserved for the user |
| RQ4b-6 (optional G6 sensitivity) | Deferred, optional, not committed | Not opened. It may start only as a new, separately authorized stream |

Closure reconciliation on 2026-10-24 (read-only):

- All three production receipts have status `PASS`, and their SHA-256 values
  equal the values above.
- The 160 source-run, Gate C and bootstrap files, and the 20 `test_evaluation`
  files, still equal their pre-Test snapshots. The Test run holds 22 files and
  an empty `failures/` directory.
- The Notebook 12 self-check reconciliation of its printed key values against
  the stored artifacts reports exact equality.

Limitations carried into any use of these results:

- Threshold calibration and Gate C lambda selection share the complete
  Validation partition, so the Validation values are optimistic.
- The replicated lambdas 0 and 0.01 have only three seeds. Lambdas 0.03, 0.1
  and 0.3 are single-seed (seed 42) screening arms and have no across-seed
  summary.
- Within-seed paired-bootstrap intervals and across-seed summaries are
  separate uncertainty views and are not pooled.
- The `training-unstable beyond selected epoch` label applies to lambda > 0
  units. Stability status is `not_assessed`.
- The real-data capability probe has not been fitted. Capability remains
  `capability_inconclusive`.
- `config/dataset_b/rq4b_adversarial.yaml` still carries its pre-execution
  `test_evaluation.execution_status: not_run` and `authorization_status`
  fields. They were deliberately left unchanged, because the Test manifest pins
  the config SHA-256 and the package validator checks it. This register
  supersedes those fields.
- The outputs describe approval-prediction behaviour for the documented HMDA
  cohort only. They are not default-risk, creditworthiness, legal-compliance,
  discrimination or causal evidence.

Process note: Tasks 4–7 (Test launch, package validation, Notebook 12 and
this closure) ran during a temporary handover from Codex. The read-only
verifier agent checked Tasks 4–6 (`PASS`). This closure is checked by the
same agent before acceptance. Per the handover note, Codex will re-check this
work after its quota resets. That re-check is not recorded here.

### RQ4b-3 applicant-sex supplementary evaluation — 2026-10-24

The evaluation-only applicant-sex supplementary run
`rq4b3_sex_supplementary_20261024_02` is `COMPLETE`; its package validation is
`PASS`. It is published separately at
`artifacts/dataset_b/rq4/rq4b/sex_supplementary/` and does not modify the
closed production Test namespace.

- Inputs were the nine saved Test predictions in
  `rq4b3_test_20261024_01`, their corresponding saved Validation predictions,
  locked thresholds, and the existing 1,000-replicate Test bootstrap index.
  The run pins the Test manifest SHA-256
  `d05fc298...e7595a4`, Test receipt SHA-256
  `2f7c1ae8...e64632c4`, package receipt SHA-256
  `d20fa9b1...f1e5800a`, and bootstrap SHA-256
  `2ee377b6...84230fee1`.
- The attribute is `applicant_sex`, using the RQ3 mapping
  `1=Male`, `2=Female`, `3=Information not provided`, `4=Not applicable`,
  and `6=Applicant selected both male and female`. EOD uses the fixed
  Male/Female roster; coverage tables retain all five groups.
- All nine arms and six same-seed nonzero-lambda comparisons are present.
  Each sex-EOD delta bootstrap has 1,000 valid replicates and zero invalid
  replicates. The package validator recomputed every output from saved
  predictions with 0 local fits, 0 refits and 0 inference calls.
- Execution status, exported metric values, and research interpretation are
  separate. This status entry records no effectiveness, support, policy, or
  research conclusion.
- The earlier supplementary namespace run `_01` remains immutable but is not
  the accepted supplementary evidence because its manifest did not list the
  Gate C and source-run pins directly. Run `_02` supersedes it for this
  supplementary package; no source or production artifact was changed.
- Reporting: `notebooks/dataset_b/final/research_questions/13_rq4b_applicant_sex_supplementary_results.ipynb`
  (SHA-256 `f75e56a8...cdd1d7b`) presents run `_02` only and is separate from
  the closure Notebook 12. It has 11 code cells with execution counts 1–11 in
  order, no saved error or stderr output, and five embedded figures. It
  performs no file writes, fit or inference. Its research-interpretation
  section is reserved for the user.

### RQ4b-3 parallel evaluation and training amendment — 2026-10-21

The user-approved RQ4b-3 amendment supersedes both the original
18-development-unit protocol and the temporary 12-fit priority policy below.
The amended execution uses two explicitly linked, non-overlapping run identities.
`rq4b3_adv_20261019_02` is the immutable Track A source run for the retained
seed-42 artifacts and remains on its historical v2 code lineage.
`rq4b3_adv_20261021_01` is the immutable v3 Track B child run that contains the
two seed-43 units. The completed seed-44 continuation uses the separate v4
identity `rq4b3_adv_20261022_01`. The maximum development
scope is nine units: five retained completed seed-42 units at lambdas
`[0, 0.01, 0.03, 0.1, 0.3]`, plus exactly four approved additional units:
`(lambda=0, seed=43)`, `(lambda=0.01, seed=43)`, `(lambda=0, seed=44)` and
`(lambda=0.01, seed=44)`. Lambda 1 is removed from the execution queue;
lambdas 0.03, 0.1 and 0.3 remain screening evidence at seed 42 only. No
additional lambda, seed, architecture search or experimental method is opened.

Track A evaluates the five frozen completed seed-42 artifacts for EOD3, paired
same-seed delta EOD3 where defined, Balanced Accuracy cost, available
diagnostics, fairness--utility evidence and training stability. It must not
refit, rerun inference, regenerate or overwrite completed checkpoints,
predictions, epochs, receipts or manifests. Artifact integrity and Gate B
execution evidence for the five units have now been directly checked and are
`PASS`. Formal Gate C evaluation outputs are now published by the separate
evaluation-only aggregation run `rq4b3_gatec_20261024_01`; see the Gate C
aggregation execution section below.

Track B is limited to the four new units above and Gate A is `PASS`. It does
not depend on Track A completion, EOD3 results or fairness acceptance. Gate A
does not require EOD3; it requires the amended configuration and implementation
to agree, correct run/seed/lambda assignment, safe nonconflicting paths and
write ownership, overwrite protection and no unresolved critical correctness
or lineage issue. The v4 input validator and runner preserve the sparse
nine-unit design while restricting the current production identity to the two
seed-44 units. Cross-run aggregation was blocked until Gate C lineage
validation; it has since run in `rq4b3_gatec_20261024_01` (below).

All four additional Track B units are complete. The two seed-43 units remain
in the immutable v3 child run and the two seed-44 units are published in the
immutable v4 continuation. Gate B is `PASS` for the complete nine-unit sparse
design. No automatic retry is authorized. Cross-run Gate C evaluation and
Validation-only selection have since been executed in
`rq4b3_gatec_20261024_01` (below). No final-fit is authorized. The later
user-selected Test output contract and cross-run implementation are complete.
Production Test evaluation and production package validation have since each
run once in `rq4b3_test_20261024_01` (see the Test evaluation and package
validation execution sections below).

Gate B requires each new unit to complete five CV folds, one full-Train refit,
development-validation, integrity and prediction-count checks, and artifact
and receipt validation. Gate C required evaluation evidence for every retained/new
development unit, EOD3, paired same-seed delta EOD3, BA cost, EOD6 safety
audit, seed variability, training stability/integrity, explicit statistical
limitations and a separately approved selection policy suitable for three
seeds; those aggregation requirements are now complete. The former 4/5 rule is
not satisfied by three seeds and must not be silently replaced with 2/3. The
Test contract below exports neutral numbers and assigns no replacement rule or
research conclusion. That contract-recording task did not open Test. Test
was later executed once, package-validated and reported (see the RQ4b-3
closure above). Execution completion, statistical evidence, fairness effectiveness, predictive
utility and training stability remain separate statuses.

### RQ4b-3 nine-unit Gate B closure — 2026-10-24

The amended nine-unit development roster is `COMPLETE`, and Gate B is `PASS`
on direct production-artifact evidence. The accepted roster is seed 42 at
lambdas `[0, 0.01, 0.03, 0.1, 0.3]`, seed 43 at lambdas `[0, 0.01]`, and seed
44 at lambdas `[0, 0.01]`. Every unit has one published development stage with
five completed CV folds and one completed fresh full-Train refit, plus one
matching development-validation stage with 43,995 ordered Validation
predictions. Successful fit accounting is 45 CV fits and nine refits.

All 18 stage receipts are `PASS`. Receipt-declared output hashes, prerequisite
links, unit identities and run-manifest lineage were checked directly. Each
Validation file has the required five-column schema, 43,995 unique nonblank
`record_id` values in the same order, binary labels, complete audit groups and
finite probabilities in `[0,1]`; every ordered-ID digest is
`f5331875e449fc2f940de5a845483d5ea0830ae2b5a8501df98e9814a2fa5875`.
The seed-44 manifest's pinned source-manifest, source-receipt and retained
seed-43 failure-receipt digests also match the immutable source runs.

The seed-44 control selected/refit epoch 120 and published Validation threshold
0.753. The seed-44 `lambda=0.01` unit selected/refit epoch 17 and published
threshold 0.766. Its five-fold mean Validation CE first exceeds 1 at epoch 32
and later reaches approximately `8.34e8` while remaining finite. Its 17-epoch
fresh refit remains in the normal finite loss range, retains finite predictor
and adversary parameters and passes exact prediction reload. Together with the
same observed pattern for seeds 42 and 43, this is a required
`training-unstable beyond selected epoch` Gate C limitation, not a Gate B
artifact failure.

This closure establishes development execution, validation coverage, integrity
and lineage only. It does not establish mitigation effectiveness. The later
evaluation-only aggregation run `rq4b3_gatec_20261024_01` pins all three
immutable source runs. Its locked Validation policy selects only replicated
lambdas 0 and 0.01 by mean matched BA eligibility (>= -0.01), then minimum
mean EOD3 with exact smaller-lambda ties; 0.03/0.1/0.3 are screening-only and
EOD6 is an audit. The subsequent user authorization of the Test output
contract is recorded below; it does not retroactively change this Gate B
closure or the stored Gate C artifact.

### RQ4b-3 Gate C aggregation execution — 2026-10-24

Production Gate C aggregation run `rq4b3_gatec_20261024_01` is `COMPLETE`.
It was executed once through the official evaluation-only path
`python -m src.dataset_b.mitigation.rq4b_runner --stage gate_c_aggregation
--run-id rq4b3_gatec_20261024_01 --confirm-production` (exit 0, empty
stderr). Before execution the run directory did not exist and the read-only
source-pin test passed. The run is published at
`artifacts/dataset_b/rq4/rq4b/gate_c_aggregation/rq4b3_gatec_20261024_01/`
with no leftover temporary tree.

Direct read-only verification:

- `read_stage` on stage `gate_c_aggregation` returns `PASS` (mode
  `evaluation_aggregation`).
- The manifest `source_pins` and `source_run_ids` equal the contract pins.
  `evaluation_only`, `prohibit_fit` and `prohibit_inference` are true.
  `test_status` is `blocked_pending_separate_methodological_gate`. The
  manifest config and code SHA-256 values equal the current files.
- Receipt counts: 0 CV fits, 0 refits, 0 probe fits, empty predictions, and
  source fit accounting of 45 CV fits and nine refits. The receipt has 18
  source receipt inputs.
- Independent re-hashing confirms that all three source manifests and all 18
  source receipts still equal their pins.
- Every unit was evaluated on the same 43,995-row ordered Validation
  population (digest `f5331875...fa5875`). The fail-closed check that the
  recomputed threshold exactly equals the stored threshold passed for all
  nine units.

Validation-only selection result (`gate_c_selection.json`, status
`selected`): **`selected_lambda = 0.0`** (`lambda_zero_selected: true`).

| lambda | seeds | mean matched delta BA | mean EOD3 | eligible |
|---|---|---|---|---|
| 0 | 42, 43, 44 | 0.000000 | 0.099633 | yes |
| 0.01 | 42, 43, 44 | -0.005011 | 0.102641 | yes |

Both candidates meet the BA eligibility bound (>= -0.01). Lambda 0 has the
lower mean EOD3, so the policy selects it. Per seed, lambda 0.01 has a lower
EOD3 than its matched control for seeds 42 and 43, and a higher one for seed
44. The selection artifact also records the nine unit rows, the
screening-only lambdas `[0.03, 0.1, 0.3]` and the EOD6 role `audit_only`.
It records `arbitrary_two_of_three_rule_used: false`.

These are separate statuses:

- Gate C aggregation execution: `COMPLETE`.
- Mitigation effectiveness: **not established**. On Validation, the
  replicated adversarial candidate (lambda 0.01) did not lower mean EOD3
  relative to the matched control.
- At Gate C publication, Test was blocked on its separate methodological gate
  and was not opened. No final-fit, Test access or package stage was run. The
  later Test output-contract authorization is recorded below and does not
  alter the stored Gate C receipt.

Limitations:

- Threshold calibration and lambda selection share the complete Validation
  partition, so the selection is optimistic.
- Only three seeds are available.
- The `training-unstable beyond selected epoch` label applies to every
  lambda > 0 unit. Its evidence for lambdas 0.03, 0.1 and 0.3 was not
  re-verified in this task.
- The results are approval-prediction fairness diagnostics only. They are not
  default risk, creditworthiness or causal/legal discrimination evidence.

The earlier Gate C post-run test closure was `PASS` at 296 passed and 20
legacy post-Gate-C skips. Those skips are superseded by the implemented
cross-run Test/package route below. The three Gate C tests that previously assumed the production
run did not exist now snapshot its existence and file SHA-256 values and
confirm that unauthorized-root checks, source-pin verification and the
synthetic CLI path neither create nor change the published production run.

### RQ4b-3 production Test evaluation execution — 2026-10-24

Production Test evaluation run `rq4b3_test_20261024_01` is execution
`COMPLETE` with stage receipt `PASS`. It was launched once, under the user's
Task 4 contract, through the official route
`.venv-realmlp-gpu\Scripts\python.exe -m src.dataset_b.mitigation.rq4b_runner
--stage test_evaluation --run-id rq4b3_test_20261024_01 --confirm-production`
(exit 0; started 2026-10-24T08:51:50Z, finished 2026-10-24T09:00:15Z; stderr
held only library/progress messages on the CUDA RTX 5050 device and the
existing quality-gate note `Dataset B LTV >1000% retained for 11 rows`, once
per inference). No retry
occurred. The run is published at
`artifacts/dataset_b/rq4/rq4b/test_evaluation/rq4b3_test_20261024_01/`
with no leftover `.tmp_*` tree and an empty `failures/` directory.

Preflight before launch (read-only): the run path did not exist; code, config
and plan/contract SHA-256 values equalled the handoff pins; Gate C manifest,
receipt and selection digests matched; three source manifests, all 18
model/Validation receipts and all nine estimator/preprocessor hashes matched;
the nine arms resolved to nine unique artifacts; the RQ1 bootstrap index digest
`2ee377b6...30fee1` and its manifest matched (1,000 replicates, 43,996 Test
rows); `cuda:0` was available in the GPU environment.

Direct post-run verification (read-only):

- Receipt `stages/test_evaluation/receipt.json` (SHA-256
  `2f7c1ae8b31d8df82b9342182cb40fb9b3c18aa783a31ff53eeeaf54e64632c4`):
  status `PASS`; 18 declared outputs, all re-hashed equal; 22 lineage inputs;
  0 CV fits, 0 refits, 0 probe fits; source accounting 45 CV fits / nine
  refits. Run manifest SHA-256
  `d05fc2980986452428b5fc656aaba89c8bad698923cab7da67f5ff241e7595a4`;
  its code and config digests equal the current files, its Gate C pins equal
  the locked values and it records the nine locked Validation thresholds.
- `execution_status.json`: `completed`, reason `none`, inference count 9,
  fit/refit/CV counts 0, 43,996 Test rows.
- All nine saved Test prediction files have 43,996 unique rows in one shared
  order whose ordered-ID digest equals the locked bootstrap manifest
  (`47dc6722...7c081d`), with finite probabilities in `[0,1]`.
- Output roster complete: nine arm rows, 81 group rows (no undefined TPR/FPR),
  six matched-delta rows, 60 decision-change rows, 6,000 bootstrap replicate
  rows with 1,000/1,000 valid replicates for every comparison and quantity
  (delta BA, EOD3, EOD6), 18 CI rows, nine across-seed rows (lambdas 0 and 0.01
  and their matched deltas only) and 27 generalization-gap rows; all metric,
  delta, CI and gap values are finite.
- Pre/post SHA-256 snapshots of all 160 files in the three source runs, the
  Gate C namespace and the bootstrap index/manifest are identical.

Separate statuses: production Test execution `COMPLETE`; production package
validation `PASS` (section below); notebook/report **not created**. Metric
values are recorded only in the published files. Research interpretation,
mitigation effectiveness and any conclusion remain with the user and are not
established by this execution.

### RQ4b-3 production package validation execution — 2026-10-24

Production package validation for `rq4b3_test_20261024_01` is `PASS`. It ran
once through
`.venv-realmlp-gpu\Scripts\python.exe -m src.dataset_b.mitigation.rq4b_runner
--stage package_validation --run-id rq4b3_test_20261024_01 --confirm-production`
(exit 0, empty stderr; 2026-10-24T09:29:47Z to 09:33:26Z). No retry occurred.
The route constructs no backend and performs no fit or inference. It re-verified
the manifest identity, code/config/protocol lineage, output schema, Gate C
pins, three source manifests, all 18 model/Validation receipts, the nine-arm
model/threshold lineage, the bootstrap pin, the receipt input/output roster
and per-file prediction records. It then recomputed every locked table and
`execution_status.json` from the nine saved Test predictions and the source
Validation predictions, and compared each stored file to its recomputation.

Evidence:

- Stage `stages/package_validation/` holds `receipt.json` (SHA-256
  `d20fa9b13cd4f9b9e5e52e4c387f49a63791f4c95af3fa9903caecdbf1e5800a`; status
  `PASS`; input `test_evaluation` receipt `2f7c1ae8...32c4`; 0 CV fits, 0
  refits, 0 probe fits, no predictions; source accounting 45 CV fits / nine
  refits) and `package_report.json` (SHA-256 `02af1bc6...db32403`;
  `inference_performed: false`, `recomputed_from_saved_predictions: true`,
  local fit counts 0, nine verified outputs). `failures/` is empty.
- The built-in gate compares stored and recomputed files byte for byte. A
  separate read-only comparison under the locked `split_and_evaluation.md`
  section 7 tolerances (point metrics `atol = 1.0e-12`; replicate values and
  CI endpoints `atol = 1.0e-14`; identifiers, counts and validity flags exact)
  found identical shapes and a maximum absolute difference of 0 for every
  table: 9 arm, 81 group, 6 delta, 60 decision-change, 6,000 replicate, 18 CI,
  9 across-seed and 27 gap rows. Every seed, arm, metric and replicate in the
  locked roster is present.
- All 20 `test_evaluation` files are unchanged. The run gained only the two
  package files. The 160 source-run, Gate C and bootstrap files equal their
  pre-Task-4 snapshot.

Notebook/report creation and all research interpretation remained pending
separate user authorization at package-validation time; the reporting notebook
is recorded below.

### RQ4b-3 reporting notebook — 2026-10-24

Reporting-only `notebooks/dataset_b/final/research_questions/12_rq4b_adversarial_results.ipynb` is
`COMPLETE` (executed and validated; SHA-256
`2876b1aed5f8621f4656b0219bceb1967e4b772059bcc07e9acd407a04ae2032`). It
reads only the accepted Gate C stage outputs, the Test run manifest, the
`test_evaluation` tables and receipt, the `package_validation` receipt and
report, and `rq4b_adversarial.yaml` for locked pins. It does not read raw data,
splits, model/preprocessor files or bootstrap indices. Prediction CSVs are
hashed for receipt integrity and never parsed. It performs no fit, inference,
bootstrap or metric recomputation.

It presents the following:

- package integrity and lineage;
- execution status and fit accounting;
- the nine-arm roster with thresholds and receipt pins;
- Validation results by lambda and seed, with the stored selection record and
  training-stability labels;
- Test BA/EOD3/audit EOD6 by arm and seed;
- G3/G6 group TPR/FPR and counts;
- six matched deltas with 95% paired bootstrap intervals and the
  6,000-replicate evidence;
- decision changes;
- stored across-seed summaries;
- Validation-to-Test gaps;
- a seed-42 screening-arm section;
- a key-value reconciliation dump.

The notebook ends with an interpretation boundary and an empty, user-owned
research-interpretation section. Code assigns no pass/fail, effectiveness,
support or recommendation label.

Validation (all `PASS`):

- Built from a generator into a new file (it refused to overwrite) and
  executed top to bottom with `nbclient` (kernel `python3`, Anaconda Python
  3.13.9, pandas 2.3.3, matplotlib 3.10.6). It has 15 code cells with
  execution counts 1-15 in order, no saved error and no stderr output, and
  embeds five figures.
- In-notebook assertions pass for the following:
  - every stage file matches its receipt;
  - the Gate C digests match the locked pins;
  - the Test manifest Gate C and bootstrap pins match the configuration;
  - the arm roster equals the locked contract;
  - each table has the expected row count;
  - replicate validity counts equal `bootstrap_summary.csv`;
  - the derived Validation means equal the stored Gate C candidate means;
  - the stored gap values equal Gate C units and `arm_metrics.csv` within
    `atol = 1e-12`.
- An implementer self-check script (AST scan; not an independent review) found
  imports limited to pathlib, hashlib, json,
  platform, sys, matplotlib, numpy, pandas, yaml and IPython. It found no fit,
  predict, transform, load, write, save, delete or subprocess call. The only
  flagged token, `estimator`, is the displayed lineage column
  `estimator_sha256`.
- An implementer self-check reconciliation, which parses the stored artifacts
  separately from the notebook, compared the printed key values and found
  exact equality for the Gate C selection, execution status,
  package report, nine arm metrics and thresholds, and 18 delta/CI/valid-count
  entries.
- `nbconvert --to html` renders the notebook. Its only warning is missing alt
  text on the five images.
- All 160 source-run, Gate C and bootstrap files and all 22 Test-run files are
  unchanged.

Research interpretation and any conclusion remain with the user.

### RQ4b-3 cross-run Test implementation — 2026-10-24

Status: **`IMPLEMENTED / SYNTHETIC VALIDATION PASS`**; the production run
recorded above is the only production execution of this route.
The cross-run Test output contract is implemented for the user-selected nine
arms. Production Test data was not read, production inference was not run, no
fit/CV/refit occurred, and no production Test run or artifact directory was
created by this implementation task.

The reserved Test run is `rq4b3_test_20261024_01` under the separate namespace
`artifacts/dataset_b/rq4/rq4b/test_evaluation/`. Its user-selected roster is
exactly all nine saved development arms: lambdas 0, 0.01, 0.03, 0.1 and 0.3 at
seed 42, plus lambdas 0 and 0.01 at seeds 43 and 44, for at most nine
saved-model Test inferences. Lambda 0 is the stored Gate C selection, lambda
0.01 is the replicated comparator, and the screening lambdas are explicitly
included at seed 42 for complete numerical reporting.

The contract pins the Gate C manifest
`5d072d7ff094817b8a281d2ea5b6527682678f97d09fa538b88013aae218931b`
and receipt
`14f602bb0ef59d7cbfb22e8b5fbdaf6604e5a90c295370bac212b965e1448f13`,
all three source-run manifests, and the model/Validation receipt pairs for all
nine arms (18 receipts). It also pins the locked RQ1 Test bootstrap index at SHA-256
`2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1`;
regeneration is prohibited.

Required neutral outputs are per-arm BA/EOD3/audit-EOD6 and group TPR/FPR plus
counts for all nine arms; every same-seed nonzero-lambda-minus-lambda-0 delta
BA/EOD3/EOD6; overall and per-group decision changes; 1,000 paired bootstrap
rows and CI endpoints for each of six matched comparisons (6,000 rows total);
three-seed mean/sample-SD/min/max/range for the replicated lambdas 0 and 0.01
and their matched deltas only; and Test-minus-Validation gaps for all nine
arms. The single-seed screening lambdas receive no across-seed summary. Exact
file names and columns are locked in `config/dataset_b/rq4b_adversarial.yaml`.

Execution status, metric values and research interpretation are separate.
Machine code may emit operational status and neutral numbers only. It may not
assign research pass/fail, mitigation effectiveness, strong support,
recommendations or a narrative conclusion; the user owns those decisions.
Implementation and synthetic validation are complete. The Test route verifies
the three exact Gate C hashes (manifest, receipt and selection output), three
source manifests, all 18 model/Validation receipts, model-file hashes, locked
thresholds, ordered Test IDs and the bootstrap digest before atomic
publication. It deduplicates inference by artifact, records local fit counts
as zero and source accounting as 45 CV fits/nine refits. The package validator
performs no inference and recomputes every metric and bootstrap output from
saved Test predictions before publishing its PASS receipt.

Validation is `PASS`: the three required RQ4b test files report **321 passed,
0 skipped, 0 failed**. Compile checks passed. The tests cover valid three-run
lineage, Gate C pin drift, model/receipt drift, missing arm/seed, Test ID/order
mismatch, bootstrap digest drift, zero local fitting, nine non-duplicated
inferences, immutable source trees, failed-run nonpublication, package
recomputation and forged output/receipt rejection. Synthetic runs were confined
to `scratch/rq4b_implementation`; direct read-only production-lineage checking
confirmed all nine arms and 18 receipt pins without creating
`rq4b3_test_20261024_01`.

Production launch, Test access/inference and production package validation
have since each been executed once (sections above). Notebook reporting and
all research interpretation remain pending separate authorization/execution.
Execution success does not determine a research conclusion.

### RQ4b-3 Gate C aggregation implementation readiness — 2026-10-24

*Historical (superseded by the execution section above).* At readiness,
production Gate C aggregation had not been executed. No `gate_c_aggregation`
run, selection artifact or Gate C result existed, and no lambda had been
selected.

The contract `gate_c_aggregation` section now pins all three immutable source
runs exactly: run IDs, manifest SHA-256 for each run (including the seed-44
manifest `f6ebf429...a53249`), all 18 development/development-validation
receipt SHA-256 values covering 45 CV fits and nine refits, and the ordered
Validation population (43,995 rows, digest `f5331875...fa5875`). The seed-42
and seed-43 pins must equal the existing continuation source evidence.

The official path is `--stage gate_c_aggregation --run-id
rq4b3_gatec_20261024_01 --confirm-production`. It constructs no backend and
cannot fit, infer or open Test. Before anything is created it rejects a missing
source run, source ID/order or mode mismatch, manifest or receipt digest
mismatch, changed output, missing/extra/duplicate/unpaired development unit,
Validation population mismatch, Validation ID ordering mismatch and pinned
population mismatch. It then publishes one run by a single directory rename;
a failure leaves no run, temporary tree or partial manifest. The locked
selection policy is unchanged. In-run `selection` is now explicitly inactive
under the nine-unit contract.

Historical validation at this readiness point was `296 passed, 20 skipped`.
The later cross-run Test implementation above replaces those 20 legacy skips;
its current closure is 321 passed with no skips.

### RQ4b-3 Track B seed-43 Gate B closure — 2026-10-22

Both currently assigned seed-43 development units are `COMPLETE`, and their
per-unit Gate B checks are `PASS`. In child run `rq4b3_adv_20261021_01`,
`(lambda=0, seed=43)` and `(lambda=0.01, seed=43)` each have a published
development stage with five completed CV folds and one completed full-Train
refit, followed by a published development-validation stage with 43,995
ordered Validation predictions. Their receipts are `PASS`; declared output
hashes, upstream receipt links, unit identities, prediction counts and
run-manifest code lineage were directly checked. Both Validation files have
the expected schema, unique `record_id` values, binary labels, finite
probabilities in `[0,1]`, and ordered-ID digest
`f5331875e449fc2f940de5a845483d5ea0830ae2b5a8501df98e9814a2fa5875`.
At the seed-43 closure audit, the then-current v3 configuration and all five
code files covered by the child-run manifest matched its stored hashes. The
later v4 continuation intentionally has new config/code lineage and never
reopens that v3 run. No temporary stage remained at closure.

The successful `(lambda=0, seed=43)` attempt selected and refit epoch 118; its
Validation threshold is 0.766. Its earlier failed attempt remains unchanged at
`failures/dev_l0_s43.ca0ec0704255402b9f2a7fb8779f57fa/`, with two CV starts,
one completed CV fit, one partial CV fit and no refit. That failed compute is
accounted separately from the successful published five-CV-plus-one-refit
attempt and was not an automatic retry. The successful `(lambda=0.01,
seed=43)` attempt selected and refit epoch 17; its Validation threshold is
0.754. It has five completed CV folds, one completed refit, no partial fit and
no prior failed attempt.

The positive-lambda 256-epoch CV trajectories show a reproducible stability
limitation rather than an artifact-completion failure. For seeds 42 and 43,
the exact minimum of the five-fold mean Validation CE is epoch 17, while mean
Validation CE first exceeds 1 at epoch 34 and later reaches approximately
`1.11e8` and `1.20e8`, respectively. Predictor loss and adversary loss also
become extremely large while remaining finite; the signed objective
`L_prediction - lambda * L_adversary` becomes strongly negative. Each fresh
full-Train refit executes only the selected 17 epochs, remains in the normal
finite loss range, retains finite predictor/adversary parameters and passes
exact prediction reload. This evidence supports Gate B execution acceptance
but requires an explicit `training-unstable beyond selected epoch` limitation
in Gate C. It does not establish an implementation defect or scientific
fairness effectiveness.

At the time of the seed-43 closure, the user approved the execution order
`(lambda=0, seed=44)` -> verify its published artifacts -> `(lambda=0.01,
seed=44)`. That sequence subsequently completed and is recorded in the
nine-unit Gate B closure above. No seed-44 fit was launched by the earlier
implementation-only task. Bypassing the runner check or silently changing a
locked manifest remains prohibited. At that time selection, final-fit, Test,
package validation and cross-run aggregation were closed. Cross-run
aggregation, Test and package validation later ran (see the RQ4b-3 closure
above). No final-fit stage was run.

### RQ4b-3 seed-44 continuation implementation — 2026-10-22

The lineage-preserving v4 continuation mechanism is implemented and its
focused synthetic validation is `PASS` (`304 passed`, `20 skipped`). The active
protocol is `rq4b_adversarial_realmlp_v4_seed44_continuation`, decision revision
`rq4b3_seed44_continuation_20261022`, and the only production run identity it
accepts is `rq4b3_adv_20261022_01` with role
`track_b_seed44_continuation`. The only authorized units in this identity are
`(lambda=0, seed=44)` and `(lambda=0.01, seed=44)`.

This is a continuation, not a migration or resume. The manifests for
`rq4b3_adv_20261019_02` and `rq4b3_adv_20261021_01`, plus all published
development/development-validation receipts for their seven completed units
and the retained seed-43 failure receipt, are pinned by SHA-256 in the v4
contract. Production create/open verifies every
pinned digest before accepting the continuation. The v4 manifest records those
source identities and digests, prohibits fitted-state reuse, and keeps
cross-run aggregation blocked pending Gate C. Neither source manifest nor any
published receipt was edited.

The runner also enforces the approved order. `(lambda=0.01, seed=44)` cannot
start until the `(lambda=0, seed=44)` development and
development-validation receipts are published, their declared output hashes
verify, fit accounting is exactly five CV folds plus one refit, and the
Validation prediction count is 43,995. A source digest mismatch stops before
run creation/open; a code or config change after creation blocks resume and
requires preserving the attempted run under its existing identity. Automatic
retry remains prohibited.

No production run, preflight, capability stage, inference or fit was launched
in that implementation-only task. The later authorized execution completed in
the required order and is recorded in the nine-unit Gate B closure above. Both
published seed-44 units are now immutable and must not be rerun or reopened.

### Superseded temporary N5-to-RQ4b-3 12-fit priority policy — 2026-10-20

This subsection is retained as historical governance evidence only. It was
superseded by the 2026-10-21 RQ4b-3 amendment above and is not an active
execution or selection policy.

The user temporarily replaces the 18-development-unit gate for the current
N5-through-RQ4b-3 path with a 12-fit priority design. The original 18-unit gate
is inactive for this path unless the user explicitly reopens it later. The
12 unique development fits are:

- screening: all six locked lambdas `[0, 0.01, 0.03, 0.1, 0.3, 1.0]` at seed
  42 (six fits);
- controls: lambda 0 at seeds 43 and 44 (two fits); and
- replication: two nonzero lambdas from the screening set at seeds 43 and 44
  (four fits).

The completed lambda-0, seed-42 development fit in production run
`rq4b3_adv_20261019_02` counts toward the 12-fit total, leaving 11 fits under
this temporary policy. The two nonzero replication lambdas are not yet named
by this status entry. This is a user-approved temporary execution/selection
policy, not evidence that the remaining fits, Validation units, selection,
final arms, Test evaluation, or package validation have completed. No fit or
inference was launched by recording this decision. Existing implementation or
configuration requirements that still encode the former 18-unit gate require
separately scoped alignment before a 12-unit selection stage can execute.

For a separately authorized N5 execution, Codex reports terminal progress once
per five minutes and immediately at phase/fold completion, process exit, or
error. After 30 minutes without confirmed progress, Codex checks the actual
process state, command identity, child processes, exit state, durable-log/file
movement, conflicts, and failure evidence. RAM, CPU, and GPU utilization are
secondary telemetry and cannot alone establish a hang or justify termination.
Codex stops only for a confirmed conflict or process exit/failure, preserves
the evidence, diagnoses and reports the reason, does not retry automatically,
and does not advance to the next unit after a failure.

### Latest chat authorization and N4 interruption — 2026-10-19

This section supersedes the older user-only-fit assignment below for the
following bounded sequence only: implement terminal reporting every 120 seconds
while retaining detailed durable file logs; test and separately self-review;
then Codex executes one fresh real N4 probe attempt, verifies its artifacts and
stops before N5. The user explicitly approved this sequence in chat. No
automatic retry, N5/production launch or scientific-method change is authorized.
The current turn records the rule and handoff only; implementation and execution
are deferred to the next chat. Terminal throttling is required, not implemented
or tested yet. Start, phase transitions, completion and errors must be reported
immediately; periodic reports distinguish observation from completed progress.

The user-run `N4_real_probe_20261019_02` is interrupted, not unused or completed.
Its failure receipt records `KeyboardInterrupt` at 2026-10-19T15:09:36.724157Z
(22:09:36 UTC+07), with the last observation at epoch 86, batch 50. There are
85 completed epochs and 515 parseable JSONL events; run.log and JSONL are byte
identical (322,096 bytes each). The last heartbeat retains the epoch-85 completed
progress timestamp. No final history summary or real_probe_evidence.json was
published. The signal source is unknown; this does not establish an optimizer
defect, OOM or hang. Preserve this attempt unchanged and use a new unused attempt
after the approved code/test gates. Capability cannot be assessed from `_02`.

### N4 logging review correction — 2026-10-19

The follow-up Codex self-review found five remaining N4 logging issues in the
initial logging fix: heartbeat/completed-progress timestamp conflation, startup
log errors outside the receipt boundary, no observer during preparation or
blocked foreground phases, premature attempt-completion logging, and missing
logging-payload validation at evidence ingestion. The authorized correction
addresses these in synthetic/mock scope; evidence is in
`scratch/rq4b_implementation/LOGGING_REVIEW_FIX_20261019_01/`.
Regression: 449 passed, one Torch deprecation warning, exit 0; an additional
observer-write-failure test passed separately. Method-function AST comparisons
and outside-scope input/evaluation/runner/YAML hashes passed. This is Codex
self-review, not independent review or real-fit evidence.

N4 now distinguishes observation time from last completed progress, observes
before frozen preparation, routes initial log errors through failure handling,
seals hashed logging payloads before atomic evidence publication, and verifies
all three logging files plus epoch/history correspondence during ingestion.
`computation_completed` is not an attempt-success claim: the published evidence
is the completion marker. Observer activity does not establish optimizer
progress and cannot guarantee output during interpreter/storage stalls or a
hard kill. Background write errors surface at the next foreground check or
shutdown; they cannot forcibly abort a blocked GPU call.

The original handoff named `N4_real_probe_20261019_02`; that handoff is now
historical because the user executed and interrupted the attempt, as recorded
above. Codex has not reopened N5 or retried the failed run. Historical N4 `_01`
and the failed N5 artifacts remain unchanged and must not bypass the new lineage
and log-integrity gates. A new N4 attempt remains pending.

Codex owns current implementation, testing, self-review and reporting. Claude
has no active assignment or verifier dependency; historical authorship and
independent reviews remain unchanged. Handover approvals are given in chat.
The approved sequence is ownership synchronization (G1), synthetic/mock
ProductionBackend integration verification (G2), then context handoff (G3),
with review before each transition. A separate Codex validation pass is
self-review, not independent review. C1-C5 implementation acceptances retain
their documented evidence limits. Real-data runtime, real probe and production
execution are not authorized by this handover.

### Previous operating assignment — 2026-10-19 (superseded for the scoped N4 sequence above)

The user confirmed the next workflow in chat: Codex implements, tests, fixes
bugs, performs a separate self-review/validation pass and reports technical
acceptance; the user executes every real fit (capability probe, CV and
full-Train refit). Codex supplies stage-specific commands before execution and
checks saved artifacts after the user's run. Codex does not launch or retry
real fits. Claude has no current assignment, reviewer role or dependency;
historical authorship and independent-review evidence remain unchanged.

N1 documentation/governance, N2 real-probe entry/ingestion preparation, N3
readiness verification and the user-executed N4 real probe are retained below.
The N4 evidence was accepted under its then-current code lineage. A later
logging fix changes `rq4b_probe.py`, `rq4b_training.py` and `rq4b_runner.py`, so
the immutable N4 attempt remains historical and must not be overwritten or
used to bypass current lineage checks. No production N5 run is opened by this
status update.

N1 documentation/governance acceptance is `PASS` (Codex self-review,
2026-10-19): YAML parsing and the locked contract loader passed; comparison
against the pre-N1 working copy preserved every non-governance YAML value
and existing historical governance payload. Static inspection matched all ten
CLI stages to the operation table; targeted whitespace checks passed. No
pytest, real preflight, fit or inference was run for N1. Changes are limited to
STATUS, the neural runbook, RQ4 plan ownership/sequencing and YAML governance.

N2 implementation/self-review is `PASS` in its synthetic/mock scope. Evidence:
`scratch/rq4b_implementation/N2_real_probe_20261019_01/review.json` and
`final.txt` (433 passed, exit 0), plus two targeted ingestion tests. It did not
execute a real probe or production fit.

N3 readiness is `PASS` in `VERIFY_ONLY` scope. Evidence:
`scratch/rq4b_implementation/N3_readiness_20261019_01/{review.json,checklist.json,code_baseline.json,input_hashes_before.json,HANDOFF.md}`.
It verified 13 readiness checks, including real inputs/environment/CUDA, and
performed no fit, inference or production-run creation.

The user-executed N4 attempt
`scratch/rq4b_implementation/N4_real_probe_20261019_01/` is accepted historical
evidence under its recorded lineage: 35,267 G3 rows split 21,160/7,053/7,054;
91 completed epochs, best epoch 76 and patience 15; CE gain
0.009407937095777696 nats with 95% CI
[0.006539494433896268, 0.012237872645475325] and 1,000/1,000 valid bootstrap
replicates. This is descriptive capability evidence, not mitigation efficacy,
production training, or authorization for N5.

The RQ4b logging fix is implementation-complete and Codex-self-reviewed in
synthetic/mock scope. It adds durable terminal/file progress, per-epoch JSONL,
non-optimistic heartbeats, full CV-fold history/log retention, traceback and
last-position failure evidence, and started/completed/partial/unknown fit
accounting. The focused four-suite regression passed 283 tests with 28
deprecation warnings; no real fit or production inference ran. Evidence:
`scratch/rq4b_implementation/LOGGING_FIX_20261019_01/`. Because source lineage
changed, the original handoff required a fresh user-run N4 attempt. The latest
chat authorization above now governs the next N4 sequence; stop before N5.

### Codex handover gates — 2026-10-18

- **G1 PASS — ownership/status synchronization.** Seven authorized files were
  updated. Contract validation and 185 input tests passed. All method sections
  and historical C1-C5 acceptance payloads were preserved; 20/20 selected
  outside-scope baseline hashes matched. Evidence:
  `scratch/rq4b_implementation/CODEX_HANDOVER_1_b4f7f8b4953c49fe92ea79a5162ac79a/review.json`.
- **G2 PASS — synthetic/mock production glue verification.** The new
  `tests/test_dataset_b_rq4b_production_backend.py` exercises actual synthetic
  disk loading, index validation, preflight collection/guard, prediction
  packaging and CLI/engine argument forwarding. There were 57 integration
  tests plus 233 related regression tests (290 passed, one Torch deprecation
  warning), and 42 existing runner tests passed separately. All 27 selected
  Task 2 baseline hashes matched; production source was unchanged. Evidence:
  `scratch/rq4b_implementation/CODEX_INTEGRATION_2_adf341da88fc4db99eaebf75c67ea8e8/review.json`.
  The initial runner harness exceeded Windows path limits; an audit-local
  fixture-root override resolved that harness issue. The final pytest log
  reports 42 passed in 201.94s; its shell pipeline was interrupted after pytest
  completed, so no clean shell-exit claim is made for that invocation.
- **G3 PASS — chat/context handoff.** The approved sequence and prior chat
  decisions were reconciled with the G1/G2 evidence. Current ownership,
  acceptance limits, remaining work and approval boundaries are recorded here
  and in the context delivered in chat. This is Codex self-review, not an
  independent acceptance or a new scientific result.

G2 closes the earlier gap in synthetic/mock coverage of ProductionBackend,
index loading and CLI dispatch. It does not close real-data runtime validation:
production context/CUDA, real fit/inference/reload, frozen inputs and the real
capability probe remain unexecuted in this handover. (Real fit, inference and
reload were later executed; see the RQ4b-3 closure. The real capability probe
remains unfitted.) The historical T5 row
below describes evidence available at C5 acceptance; this section records the
later G2 evidence. Real capability remains `capability_inconclusive` while its
probe has not run. RQ4b-6 remains deferred.

The N1 operating instructions describe stage commands, paths, stop conditions
and recovery without launching them. Some failed stages block continuation of
their run because they have no retry option; do not silently redesign that
behavior. Real capability fitting and staged production execution remain
separately scoped, user-executed work. Do not launch the full grid from this
handover. The locked main-plan budget remains at most 110 CV fold fits plus
22 refits; probe fits, spike, retries and failed attempts are accounted for
separately, not hidden within that bound.

Statuses describe only the named workstream and evidence scope. A completed experiment on `temporary_training_lock_v0_2` is not a final feature-selection result or a completed research question. Integrity metadata recorded for those historical runs is `legacy evidence only`; final retraining acceptance uses direct schema, content, row-alignment, and numerical checks.

## Current final-retraining state

- The ordered 15-field `FINAL_MODEL_FEATURES` contract and 12-player
  `EXPLANATION_GROUPS` mapping are complete. Feature selection is not an open
  decision.
- The data-quality, classical preprocessing, neural preprocessing, split and
  evaluation, baseline training, and neural training documents now describe
  the final 15/19/62 workflow.
- The documentation and preprocessing policy is updated in place. Final run
  acceptance uses direct schema/content checks and one `run_id` connecting
  preprocessing, model, Validation threshold, Test predictions, and evaluation.
- Existing 12-feature preprocessors, fitted models, matrices, predictions,
  thresholds, metrics, and review packages are historical experimental evidence
  only. They must not be resumed, extended, relabeled, or reused for final
  retraining.
- The final LR run `lr_final_15f_20261002_01`, RF run
  `rf_final_15f_20261002_01`, PlainMLP run
  `plain_mlp_final_15f_20261004_03`, and RealMLP run
  `realmlp_final_15f_20261004_01` are complete through their authorized final
  chains and have `FROZEN` publication markers. XGBoost final run
  `xgb_final_15f_20261003_01` has a validated reporting package and a
  `FROZEN` publication marker. TabMmini run
  `tabmmini_final_15f_20261004_01` is directly reviewed, published, and frozen.
- Neural implementation convergence is complete for the frozen PlainMLP,
  RealMLP, and TabMmini routes. The shared boundary is raw
  input through one central quality gate, followed by normalized 15-column
  input, fold-local 19-column semantic preprocessing, and a model-specific tail.
- Git source closure is present locally and on `origin/main` for PlainMLP at
  `4378264`, RealMLP at `5d27abd`, and TabMmini at `3f47dd5`.
- All six final reporting inputs are now available under complete run lineage.
  The Dataset B RQ2 GSV/interventional Shapley--Lorenz adaptation protocol and
  its numerical-fixture specification are locked. Predictor adapters, exact
  numerical methods, the minimal production runner, and their targeted tests
  are implemented. The current RQ2 route uses CPU-only frozen inference; the
  historical GPU training and freeze evidence remains unchanged. Production
  run `rq2_cpu_production_20261009_01` and its final artifact validation pass
  for all six models. The accepted output contains the shared sample, donors,
  bootstrap indices, six model packages, 60 cross-model stability rows, and
  complete runtime evidence. The verified results notebook is
  `notebooks/dataset_b/final/research_questions/08_rq2_gsv_sl_results.ipynb`. The benchmark remains
  separate test evidence and was not imported into production. RQ3 production
  run `rq3_fairness_20261010_01`, independent validation, and executed results
  notebook are complete under the separate RQ3 contract.

| Workstream/task | Status | Outcome | Evidence | Validation | Unverified | Next gate | Updated |
|---|---|---|---|---|---|---|---|
| Dataset B split v1 | COMPLETE | Immutable Train/Validation/Test membership is available: 205,310 / 43,995 / 43,996 records. | `artifacts/dataset_b/split_v1.json`; `data/dataset_b/split_v1/{train,validation,test}.csv` | Prediction IDs and order for LR, RF, XGBoost, and RealMLP matched their corresponding split partitions. | None for the membership comparisons performed in this reconciliation. | Preserve this split for all downstream work. | 2026-09-27 |
| Final run documentation policy | COMPLETE | Active contracts and runbooks use direct schema/content/data-quality validation and one `run_id` for cross-stage linkage; no opaque content identity is an acceptance gate. | `docs/handbook/contracts/{feature_governance,data_contract,preprocessing_classical,preprocessing_neural,split_and_evaluation}.md`; `docs/handbook/runbooks/{baseline_training,neural_training,tabmmini_training}.md` | The final configurations, validators, runners, reporting/review modules, and frozen artifacts now implement the policy for all six final model inputs. Neural publication source enforces working review, atomic rename, post-move validation, then `FROZEN`. | RQ2/RQ3 decisions remain outside this implementation claim. | Preserve the final runs and apply the same publication order to any separately authorized future run. | 2026-10-05 |
| LR baseline — experimental RQ1 evaluation | COMPLETE | Full-Train fit, Validation threshold `0.747`, and Test evaluation exist for feature lock v0.2. This is historical experimental evidence, not final-feature-contract evidence. | `notebooks/dataset_b/v1/01_linear_model_v1.ipynb`; `artifacts/dataset_b/lr_baseline_v1/` | Historical notebook outputs were inspected; 43,995 Validation and 43,996 Test IDs/order matched split, with 1,000 bootstrap rows. | No separate persisted CV OOF artifact; CV evidence is in notebook output. No clean-kernel rerun was established. | Preserve as historical evidence only; retrain from regenerated final preprocessing and fresh state. | 2026-10-02 |
| RF baseline — experimental RQ1 evaluation | COMPLETE | Full-Train fit, Validation threshold `0.662`, and Test evaluation exist for feature lock v0.2 as historical experimental evidence. | `notebooks/dataset_b/v1/02_random_forest_v1.ipynb`; `artifacts/dataset_b/rf_baseline_v1/` | Historical CV/OOF artifacts are present; Validation/Test IDs and order matched split, with 1,000 bootstrap rows. | Notebook execution counts are not a clean-kernel continuous sequence. No clean-kernel rerun was established. | Preserve as historical evidence only; retrain from regenerated final preprocessing and fresh state. | 2026-10-02 |
| XGBoost baseline — experimental RQ1 evaluation | COMPLETE | Full-Train fit, Validation threshold `0.744`, and Test evaluation exist for feature lock v0.2 as historical experimental evidence. | `notebooks/dataset_b/v1/03_xgboost_v1.ipynb`; `artifacts/dataset_b/xgb_baseline_v1/` | Historical CV/OOF artifacts are present; Validation/Test IDs and order matched split, with 1,000 bootstrap rows. | Notebook execution counts have gaps; preprocessing and run library versions differ in the CV evidence. No clean-kernel rerun was established. | Preserve as historical evidence only; retrain from regenerated final preprocessing and fresh state. | 2026-10-02 |
| LR final preprocessing — LR-00 through LR-06 | COMPLETE | The approved imbalance setting, final 15/19/62 contract, quality gate, LR profile, run-scoped exporter, persisted production CV artifact, and pre-train artifact acceptance are complete for `lr_final_15f_20261002_01`. No LR estimator fit is implied by this named preprocessing stage. | `config/dataset_b/{feature_set_v1,model_protocol_v1}.yaml`; `src/dataset_b/preprocessing/`; `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/lr_final_15f_20261002_01/` | The persisted manifest, five `rows.npz` payloads, and all 20 LR/tree matrices were reloaded. The artifact passed 15/19/62, quality and normalization audits, CSR `float64`, finiteness, row/label alignment, zero fold overlap, and exact `1/1` fold-validation coverage across 205,310 Train rows. The legacy root was not overwritten. | The previously reported 76-test suite was not rerun during production materialization; later LR stages are recorded separately below. | Preserve this frozen preprocessing lineage for the completed LR run. | 2026-10-02 |
| LR final training — LR-07 | COMPLETE | Train-only five-fold CV/OOF and a fresh full-Train LR fit are complete under the existing `lr_final_15f_20261002_01` lineage. The fixed protocol configuration was used without search; no legacy model, 66-column matrix, historical threshold, Validation, or Test input was used. | `data/preprocessing/dataset_b_v1/cv_preprocessing_v2/lr_final_15f_20261002_01/`; `artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/` | All five folds contain 164,248 fit rows and 41,062 OOF rows. Fold ROC-AUC values are 0.8155975959, 0.8151250142, 0.8188437842, 0.8145647412, and 0.8185850994; mean/std are 0.8165432470 / 0.0018045364 and OOF ROC-AUC is 0.8165073376. Direct review matched all 205,310 unique ordered Train IDs and reproduced OOF ROC-AUC. Checkpoint inspection confirmed the fitted balanced LR pipeline, preprocessor, estimator, 62 inputs, CSR `float64`, and zero convergence warnings without invoking prediction. | Source version-control closure occurred after model execution by explicit decision; no pre-execution commit provenance is claimed. | Preserve the canonical final bundle; do not overwrite or rerun this run ID. | 2026-10-03 |
| LR final holdout, review, and freeze — LR-08 through LR-10 | COMPLETE | Existing LR evidence is frozen under `lr_final_15f_20261002_01` without retraining, reinference, threshold change, or bootstrap regeneration. | `artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/`; `notebooks/dataset_b/final/models/01_linear_model_final.ipynb`; LR sources under `src/dataset_b/` | Direct LR-10 review matched 43,995 ordered unique Validation IDs and 43,996 ordered unique Test IDs, reproduced the unique threshold `0.493`, point metrics, all 1,000 locked-index bootstrap rows, percentile CIs, and the common reporting schema. Publication recorded zero fit calls, zero prediction calls, no threshold change, and no bootstrap regeneration. Acceptance used direct content and numerical checks; SHA was not a freeze gate. | No six-model comparison, RQ2, or RQ3 conclusion is implied. | Preserve the `FROZEN` final bundle and continue remaining model workstreams separately. | 2026-10-03 |
| RF final chain and freeze — RF-07 through RF-11 | COMPLETE | The existing final 15-feature RF run `rf_final_15f_20261002_01` is frozen without retraining or reinference. Five Train-only CV estimators, one fresh full-Train estimator, Validation threshold `0.712`, frozen Test evaluation, direct bootstrap recomputation, and read-only reporting are linked under the same run ID. | `artifacts/dataset_b/rf/final/rf_final_15f_20261002_01/`; `notebooks/dataset_b/final/models/02_random_forest_final.ipynb`; RF sources under `src/dataset_b/` | RF-10 directly revalidated 205,310 OOF rows with coverage 1/1, 43,995 ordered Validation rows, 43,996 ordered Test rows, the 15/19/62 contract, the frozen threshold, stored predictions, 1,000 locked-index bootstrap rows, fresh RF CIs, and the common reporting schema. Checkpoint inspection confirmed balanced 100-tree fitted estimators, 62 inputs, and no log/scaling. Publication recorded zero fit calls, zero prediction calls, no threshold change, and no bootstrap regeneration. The staged RF/shared-contract suite passed 95 tests with 7 XGBoost-only tests deselected. | Source version-control closure occurred after model execution by explicit decision; acceptance therefore rests on direct artifact/content validation rather than a pre-execution source freeze. No paired six-model conclusion, RQ2, or RQ3 completion is implied. | Preserve the final bundle and `FROZEN` marker; do not overwrite or rerun this run ID. | 2026-10-03 |
| PlainMLP preprocessing — PMLP-00 | COMPLETE | Train-only five-fold preprocessing is complete for the `plain_mlp_external_fixed_schema_ohe` CV route under `temporary_training_lock_v0_2`. | `data/preprocessing/dataset_b_v1/cv_neural_preprocessing_v1/manifest.json`; five fold directories containing 10 CSR matrices and five row-alignment payloads; `docs/handbook/contracts/preprocessing_neural.md` §§8.2, 9 | Historical validation recorded matching payload integrity. Each fold has a `164,248 × 66` train matrix and a `41,062 × 66` fold-validation matrix; all 10 matrices are CSR `float32`, have the expected shape, and contain only finite values. Each row payload partitions all 205,310 Train positions without overlap, and all saved post-`log1p` audits are finite. | No standalone retained test-run report was inspected; completion is established only within the historical 12-feature scope. | Preserve as historical experimental evidence; regenerate final preprocessing under a new `run_id` and do not reuse these matrices or fitted state. | 2026-10-02 |
| PlainMLP — PMLP-01/02 | COMPLETE | Train-only five-fold CV/OOF and the PMLP-02 configuration decision are complete within the historical 12-feature experiment. | `artifacts/dataset_b/plain_mlp_v1/{cv_results.json,cv_oof_predictions.csv,cv_training_history.json,configuration_selection.json}` | PMLP-01 reports five folds and 205,310 unique OOF IDs with finite probabilities; mean CV ROC-AUC is `0.8484776096` and OOF ROC-AUC is `0.8481142379`. The retained decision was `accept_for_full_train_refit`; artifacts state Validation/Test were not used. The current `plain_mlp_v1.yaml` supersedes that historical starting configuration for future final retraining only. | The current `04_plain_mlp_final.ipynb` is the final-run report and is not evidence for this historical row. | Preserve as historical experimental evidence only. | 2026-10-04 |
| PlainMLP — PMLP-03 | COMPLETE | A preprocessor and PlainMLP were fitted on all 205,310 Train rows under the historical `temporary_training_lock_v0_2`. | `artifacts/dataset_b/plain_mlp_v1/full_train/{checkpoints/fitted_bundle.joblib,full_train_result.json,manifest.json,COMPLETED}`; `artifacts/dataset_b/plain_mlp_v1/final_review/validation_results.json` | The historical production validator reports `PASS`; the recorded CSR `float32` matrix is `205,310 × 66` and finite. | None within the named historical scope; the bundle is incompatible with the final 15-feature contract. | Preserve as historical experimental evidence; do not resume or reuse the bundle, preprocessor, or model for final retraining. | 2026-10-02 |
| PlainMLP — PMLP-04A/04B | COMPLETE | The historical PMLP-03 bundle produced ordered Validation predictions for 43,995 rows and threshold `0.738` under the 12-feature experiment. | `artifacts/dataset_b/plain_mlp_v1/validation/{validation_predictions.csv,validation_result.json,run_manifest.json,COMPLETED}`; `artifacts/dataset_b/plain_mlp_v1/validation/threshold_selection/{threshold_candidates.csv,threshold_selection.json,threshold_manifest.json,COMPLETED}`; `artifacts/dataset_b/plain_mlp_v1/final_review/validation_results.json` | Both historical production validators report `PASS`; ordered unique IDs, zero fit calls, and one Balanced-Accuracy maximizer from 999 candidates were recorded with no Test access. | None within the named historical scope; the threshold is not valid for a newly fitted final model. | Preserve as historical experimental evidence; regenerate Validation predictions and select a new threshold for the final `run_id`. | 2026-10-02 |
| PlainMLP — PMLP-05 experimental RQ1 evaluation | COMPLETE | The retained ordered Test predictions and RQ1 metric package cover all 43,996 Test rows under historical `temporary_training_lock_v0_2`, using threshold `0.738` and 1,000 shared stratified bootstrap replicates. | `artifacts/dataset_b/plain_mlp_v1/test_evaluation/{test_predictions.csv,rq1_test_metrics.json,rq1_bootstrap_metrics.csv,run_manifest.json,COMPLETED}`; `artifacts/dataset_b/plain_mlp_v1/final_review/validation_results.json` | The historical production validator reports `PASS`; ordered unique Test IDs, the threshold, metric package, and 1,000 valid bootstrap rows were retained. | The current `04_plain_mlp_final.ipynb` reports the final 15-feature run and must not be read as historical PMLP-05 evidence. | Preserve as historical experimental evidence only. | 2026-10-04 |
| PlainMLP — PMLP-06 historical final review | COMPLETE | Tests, production validators, and bootstrap recomputation passed within the historical experimental RQ1 scope. | `artifacts/dataset_b/plain_mlp_v1/final_review/{review_manifest.json,bootstrap_recomputation_report.json,test_report.json,pytest_report.xml,pytest_stdout.txt,runtime_identity.json,validation_results.json,COMPLETED}` | The retained runtime reports 42 passed, 0 failed, 0 errors, and one non-blocking warning; the historical stages report `PASS`. Recomputing 1,000 locked replicates matched all 10 metrics under the then-approved numerical policy. | The historical notebook presentation is no longer retained at `04_plain_mlp_final.ipynb`; that path now reports the final run. | Preserve the historical artifact package; do not reuse its fitted state, predictions, threshold, or completion claim. | 2026-10-04 |
| PlainMLP final retraining — PMLP-F00 through PMLP-X09 | COMPLETE | Final PlainMLP run `plain_mlp_final_15f_20261004_03` completed the final 15/19/62 route through CUDA smoke, Train-only pilot/CV, fresh full-Train fit, one-pass Validation/Test evaluation, locked bootstrap review, reporting, recovery validation, and freeze. | `artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03/`; `notebooks/dataset_b/final/models/04_plain_mlp_final.ipynb`; `config/dataset_b/plain_mlp_v1.yaml`; source closure commit `4378264` | The common summary records five-fold mean/OOF ROC-AUC `0.8609` / `0.8605`, 205,310 OOF rows, final epoch 19, Validation threshold `0.543`, 43,996 ordered Test rows, Test ROC-AUC `0.8588`, balanced accuracy `0.7778`, and 1,000 valid locked-bootstrap replicates. Recovery evidence reports no retraining or reinference, reporting status is `COMPLETE`, all named stages are `PASS`, and `FROZEN` is present. | No RQ2 explanation output or RQ3 audit conclusion is implied. | Preserve the frozen run and use its reporting package only in separately authorized cross-model work. | 2026-10-05 |
| RealMLP — RMLP-01 through RMLP-10 experimental RQ1 evaluation | COMPLETE | Train-only CV, full-Train fit, Validation prediction/threshold `0.771`, Test/RQ1 evaluation, and final review are complete under historical `temporary_training_lock_v0_2`. This is not final-feature-contract evidence. | `notebooks/dataset_b/v1/05_realmlp_v1.ipynb`; `config/dataset_b/realmlp_v1.yaml`; upstream artifacts in `artifacts/dataset_b/realmlp_v1/`; reissued Validation/Test and RMLP-10 review in `artifacts/dataset_b/realmlp_v1_recovery_20260928/` | Historical production validators from pilot/runtime through RMLP-09 report `PASS`; 205,310 Train OOF IDs, 43,995 Validation IDs, and 43,996 Test IDs match the ordered split. The clean notebook execution has counts `1–13`, outputs for all 13 code cells, and no saved errors. RMLP-10 records 52 related tests passed. | None for the named historical RealMLP stages. The result remains scoped to the temporary feature lock and does not complete final model selection, RQ2, or RQ3. | Preserve as historical experimental evidence only; do not resume its native preprocessor, model, predictions, or threshold for final retraining. | 2026-10-02 |
| RealMLP final 15-feature run | COMPLETE | Final RealMLP run `realmlp_final_15f_20261004_01` is directly reviewed and frozen with one run lineage across runtime, Train-only pilot/CV/full-Train, Validation thresholding, Test evaluation, reporting, and closure. | `artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/`; `notebooks/dataset_b/final/models/05_realmlp_final.ipynb`; `config/dataset_b/realmlp_v1.yaml`; source closure commit `5d27abd` | CUDA/PyTabKit runtime and two-epoch smoke passed; the complete Train-only pilot locked epoch 50; five-fold OOF contains 205,310 ordered unique Train IDs with mean ROC-AUC `0.8639`; the fresh full-Train bundle contains 50 complete epochs. Validation contains 43,995 ordered unique IDs and uniquely selected threshold `0.726`. Test contains 43,996 ordered unique IDs; point ROC-AUC is `0.8632`, balanced accuracy is `0.7828`, and all 1,000 locked-index bootstrap rows and percentile intervals matched direct recomputation. Common reporting, the six-code-cell read-only notebook with four rendered SVG figures, pre-move review, atomic move, post-move review, and `FROZEN` marker passed with zero closure fit or prediction calls. The current consolidated RealMLP suite passes 12 tests. | No RQ2 explanation output or RQ3 audit conclusion is implied. | Preserve the frozen directory against overwrite; use it as eligible final RealMLP evidence for later separately authorized cross-model work. | 2026-10-05 |
| TabMmini | COMPLETE | Final TabMmini run `tabmmini_final_15f_20261004_01` completed CUDA smoke, the Train-only pilot, five-fold CV, fresh full-Train fit, one-pass Validation/Test evaluation, locked-bootstrap review, reporting, atomic publication, notebook validation, and freeze. | `config/dataset_b/tabmmini_v1.yaml`; `artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01/`; `notebooks/dataset_b/final/models/06_tabmmini_final.ipynb`; `notebooks/dataset_b/final/research_questions/07_model_total_review.ipynb`; source closure commit `3f47dd5` | CUDA smoke locked batch 256; the pilot locked epoch 55. Five-fold OOF covers all 205,310 ordered Train IDs exactly once with mean ROC-AUC `0.8572`; the fresh full-Train bundle passed exact reload comparison. Validation contains 43,995 ordered IDs and locked threshold `0.508`. Test contains 43,996 ordered IDs, ROC-AUC `0.8578`, balanced accuracy `0.7799`, and 1,000 locked-bootstrap replicates matched direct recomputation. The retained artifact review passed 25 lineage/content/numerical checks; the current source review recomputed 10 metrics across all 1,000 locked bootstrap replicates with zero fit/prediction calls, and the current targeted suite passes 33 tests. Notebook 06 is executed and read-only, notebook 07 validates all 6/6 reporting inputs, and `FROZEN` exists. | The retained direct-review JSON records the historical `threshold/` compatibility finding from publication time. Source closure `3f47dd5` reconciles the reusable validator with the authoritative `validation/threshold_selection.json` layout; the frozen artifact itself was not rewritten. No RQ2/RQ3 conclusion is implied. | Preserve the frozen run against overwrite; use its reporting package as the sixth eligible input for separately authorized cross-model work. | 2026-10-05 |
| Neural shared preprocessing convergence | COMPLETE | The central quality gate owns raw validation and `1111` normalization; model-specific pipelines consume normalized 15-column input and derive the shared ordered 19-column semantic core. PlainMLP emits fixed-schema 62-column CSR `float64`; RealMLP and TabMmini consume their native typed 19-column routes without external scaler/OHE. | `docs/handbook/contracts/preprocessing_neural.md` §§2–3, 6; `src/dataset_b/preprocessing/{neural_profiles,neural_log_transformer,realmlp_native,tabmmini_native,export_neural_cv}.py`; final PlainMLP, RealMLP, and TabMmini runs above; related tests | All three final neural runs passed their model-specific 15/19/62 or native 15/19 boundaries, central quality gate, row/order preservation, dtype/container requirements, one-pass holdout handling, reporting, and freeze gates. | None for the three final neural preprocessing routes. | Preserve all three frozen neural runs and their model-specific fitted states. | 2026-10-05 |
| Review-candidate screening | COMPLETE | Review-candidate screening is complete for all 18 raw fields: 3 keep and 15 exclude decisions reconcile across 10 blocks; both raw-versus-derived representation choices are resolved. | `notebooks/dataset_b/v1/eda/05_feature_selection.ipynb`; `docs/handbook/contracts/feature_governance.md` §§7–8 | Clean-kernel execution used the full 293,301-record cohort. All 33 code cells executed sequentially `1–33`, every code cell has output, no saved error is present, informative-only Cramér's V is `0.1735`, and all final reconciliation gates pass. No split membership or `record_id` was used. | None for the feature-screening decision. | Preserve the completed evidence; downstream configuration and preprocessing must implement the final decision without reopening screening. | 2026-10-02 |
| Final feature lock and explanation groups | COMPLETE | `FINAL_MODEL_FEATURES` contains the ordered 15 raw fields and final `EXPLANATION_GROUPS` contains 12 business players. The governance contract publishes the completed decision in place. | `docs/handbook/contracts/feature_governance.md` §§9, 11–12 | The 15 kept and 20 excluded fields reconcile to the 35-field candidate universe; both representation choices are closed. Final LR, RF, XGBoost, PlainMLP, RealMLP, and TabMmini reporting inputs implement the locked feature contract. | None for final-model feature-contract coverage. | Preserve the feature decision; do not reopen feature selection in downstream RQ work. | 2026-10-05 |
| Final preprocessing and full-model retraining | COMPLETE | All six final 15-feature model reporting packages are available and all six have `FROZEN` publication markers. | `config/dataset_b/{model_protocol_v1,plain_mlp_v1,realmlp_v1,tabmmini_v1}.yaml`; active preprocessing/evaluation contracts and runbooks; six final reporting packages connected by `notebooks/dataset_b/final/research_questions/07_model_total_review.ipynb` | Notebook 07 saved output validates 6/6 `COMPLETE` summaries and shared feature/split, ordered Test membership, and bootstrap-index lineage; a direct current marker scan confirms all six `FROZEN` files. The three neural targeted suites pass 51, 12, and 33 tests respectively. | XGBoost retains the approved exception for the independent XGB-09 bootstrap re-validation. | Preserve the six final packages; open RQ2/RQ3 only under their separate contracts. | 2026-10-05 |
| RQ1 paired-bootstrap cross-model comparison | COMPLETE | All 15 pairwise comparisons among the six frozen final 15-feature models are complete on the ordered `split_v1` Test partition using the locked shared Test bootstrap rows and the accepted ten-metric RQ1 contract. | `notebooks/dataset_b/final/research_questions/07_model_total_review.ipynb`; six frozen final Test prediction and bootstrap-metric packages; `artifacts/dataset_b/rq1_test_bootstrap_indices.npz` | The saved notebook covers `15 × 10 = 150` pair/metric rows, with exactly 1,000 shared paired replicate IDs for every comparison. Percentile 95% CIs exclude zero for 131 rows and contain zero for 19 rows. Five RF `calibration_slope` comparisons retain 1,000 paired rows but have 998 finite paired differences because two frozen RF slope payloads are invalid. Notebook JSON and all code cells validate, execution counts are sequential `1–13`, no saved error is present, and no fit, prediction, inference, or bootstrap-index generation call appears. | Evidence is metric-specific and does not authorize an absolute-best model claim, a fairness-based model choice, or any causal, legal-discrimination, creditworthiness, or default-risk conclusion. The standard `nbconvert` launcher remained unverified on this Windows host; the saved notebook completed through the verified Python 3.12.13 sequential executor. | Preserve the paired-comparison evidence. The RQ1 prerequisite for RQ4 is satisfied; RQ4 follow-up is tracked in the RQ4 row. | 2026-10-17 |
| RQ2 | COMPLETE | Production run `rq2_cpu_production_20261009_01` completed the locked six-model CPU GSV/interventional Shapley--Lorenz protocol in order for LR, RF, XGBoost, PlainMLP, RealMLP, and TabMmini. The results notebook presents only the accepted saved artifacts. | `artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01/`; `config/dataset_b/rq2_protocol.yaml`; `docs/handbook/contracts/rq2_explanation.md`; `notebooks/dataset_b/final/research_questions/08_rq2_gsv_sl_results.ipynb`; `src/dataset_b/gsv_sl/{cpu_inference,predictors,exact_methods,runner}.py`; `tests/test_dataset_b_rq2_{contract,predictors,runner,toy_fixtures}.py` | Final validation passed 91 checks with zero failures: the sample is 256 unique Validation IDs; donors use a 128-row Train pool; Test IDs are absent; the shared bootstrap index is `1000 x 256`; all six cubes are `4096 x 256`, `float64`, finite, and within `[0,1]`; every model has 24 importance rows, four bootstrap rows with 1,000 valid replicates per metric, ten checkpoint pairs, and `status.json = PASS`; runtime has six ordered rows totaling `4817.4179849` seconds; cross-model stability has 60 rows. Benchmark evidence is separate and frozen artifacts predate production. The executed notebook has 12 code cells with execution counts `1–12`, nine PNG outputs, no saved errors, and reads production tables only. | Agreement and stability are descriptive properties of the frozen models under the locked replacement game; they are not ground truth, causal validity, fairness, or proof that an explanation is correct. | Preserve the accepted run, frozen inputs, and read-only notebook boundary. | 2026-10-09 |
| RQ3 | COMPLETE | Production run `rq3_fairness_20261010_01` completed the locked six-model descriptive audit for `applicant_sex` and `derived_race` on the fixed Test partition. Approved remains the positive outcome throughout RQ1-RQ3, and the results notebook reads accepted outputs only. | `artifacts/dataset_b/rq3/rq3_fairness_20261010_01/`; `config/dataset_b/rq3_protocol.yaml`; `docs/handbook/contracts/{rq3_fairness,data_contract,split_and_evaluation}.md`; `notebooks/dataset_b/final/research_questions/09_rq3_fairness_results.ipynb`; `src/dataset_b/fairness/{metrics,audit_data,runner}.py`; `tests/test_dataset_b_rq3_{contract,toy_fixtures,audit_data,runner,production}.py`; `../Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx` | Final validation passed 24 tests with zero failures. The audit view contains 43,996 ordered Test rows; all six model statuses are `PASS`; outputs contain 588 group rows, 420 pairwise/global gap rows, 1,008 bootstrap summaries, and six runtime rows totaling 686.7488264 seconds. The runner recorded zero fit calls, zero prediction calls, and no bootstrap-index generation. Notebook 09 has eight executed code cells with execution counts `1-8`, ten embedded PNG outputs, no saved errors, and a successful 1,499,758-byte HTML render; it reads accepted RQ1/RQ3 artifacts only. | Results are descriptive model-audit evidence only. Coverage-only and denominator-ineligible results remain explicitly undefined; no causal, legal-discrimination, creditworthiness, or default-risk claim is allowed. | Preserve the accepted production directory, frozen inputs and thresholds, locked bootstrap indices, Approved-positive orientation, and visualization-only notebook boundary. | 2026-10-10 |
| RQ4 Equalized-Odds mitigation (RQ4a closed; RQ4b-3 closed; RQ4b-6 deferred) | COMPLETE | RQ4a production execution and reporting are `COMPLETE`; Codex final review C8 is `PASS`, and the manuscript-equivalent RQ3a is closed. RQ4a-3 (primary, fixed G3) run `rq4a3_postproc_20261017_01`: XGBoost and RealMLP are each `completed / stable / pass` with reason `none`. RQ4a-6 (required sensitivity, fixed G6) run `rq4a6_postproc_20261017_01`: XGBoost and RealMLP are each `completed / unstable / fail` with `target_eod_reduction_not_demonstrated`, `ba_budget_exceeded`, and `unstable_diagnostics`; this is a predeclared criterion outcome, not an execution failure, and RQ4a-6 is not promoted to primary. Notebook 11 presents both runs from their `reporting/` directories only. RQ4b technical spike R1 has C-spike PASS in synthetic two-epoch scope. RQ4b-3 method decisions are locked; T1-R1 input boundary has Codex C1 PASS and T2-R1 training/CV/refit engine has Codex C2 PASS and T3-R2 capability-probe implementation has Codex C3 PASS in synthetic validation scope (retained synthetic gate PASS; real probe not fitted); T4-R1 pure selection/evaluation logic has Codex C4 PASS and T5-R1 staged runner/reporting has Codex C5 PASS on synthetic fixtures. T1-T5 implementation gates are accepted within their documented synthetic scopes. RQ4b-3 production closure (2026-10-24): nine development units `COMPLETE` with Gate B `PASS`; Gate C run `rq4b3_gatec_20261024_01` `COMPLETE` (stored `selected_lambda = 0.0`; Validation mitigation effectiveness recorded as not established); Test run `rq4b3_test_20261024_01` `COMPLETE` with receipt `PASS`; production package validation `PASS`; Notebook 12 reports all neutral outputs. RQ4b-3 research interpretation is user-owned and not recorded. RQ4b-6 remains deferred, optional, and not committed. | `docs/handbook/RQ4_MITIGATION_PLAN.md`; `config/dataset_b/{rq4a_postprocessing,rq4b_adversarial}.yaml`; `src/dataset_b/mitigation/{inputs,metrics,postprocessing,runner}.py`; `tests/test_dataset_b_rq4a_{inputs,metrics,postprocessing,runner}.py`; `artifacts/dataset_b/rq4/rq4a/{rq4a3_postproc_20261017_01,rq4a6_postproc_20261017_01}/`; `notebooks/dataset_b/final/research_questions/11_rq4a_equalized_odds_results.ipynb`; RQ4b-3: `src/dataset_b/mitigation/rq4b_{inputs,evaluation,runner,training,probe}.py`; `artifacts/dataset_b/rq4/rq4b/{rq4b3_adv_20261019_02,rq4b3_adv_20261021_01,rq4b3_adv_20261022_01}/`; `artifacts/dataset_b/rq4/rq4b/gate_c_aggregation/rq4b3_gatec_20261024_01/`; `artifacts/dataset_b/rq4/rq4b/test_evaluation/rq4b3_test_20261024_01/`; `notebooks/dataset_b/final/research_questions/12_rq4b_adversarial_results.ipynb` | RQ4b-3: see the "RQ4b-3 closure — 2026-10-24" section, which records three `PASS` production receipts (Gate C, Test, package), unchanged source/Gate C/bootstrap snapshots, and Notebook 12 with 15 sequential cells, no saved error and exact key-value reconciliation. Governance and tooling: T2 contract delta `COMPLETE` (C2 `PASS`); T3 CPU environment/Fairlearn smoke `COMPLETE` (C3 `PASS`, run `rq4a_cpu_env_smoke_20261016_02`, Fairlearn `0.15.0` in `.venv-rq4a-cpu`); T4 pure core `COMPLETE` (C4 `PASS`); T5 runner `COMPLETE` (C5 `PASS`). Production: both runs used the CPU environment (Python 3.13.14, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, PyYAML 6.0.3, joblib 1.6.0, Fairlearn 0.15.0, recorded in each `artifact_index.json`), re-ran the input preflight, published 16 files each atomically, passed the runner package validator, and wrote no failure receipt. Each run has 43,995 Validation / 43,996 Test rows per model with exact frozen `record_id`/`y_approved`/`p_approved` lineage, baseline decisions equal to `p_approved >= frozen threshold`, unchanged decisions outside the target roster, no base-model fit or inference, and 1,000/1,000 valid replicates for target EOD, audit EOD, and BA cost (minimum 950). RQ4a-3 Test paired deltas (point [95% CI]): XGBoost delta EOD3 -0.0738 [-0.1075, -0.0246], BA cost 0.00135 [0.00013, 0.00629]; RealMLP delta EOD3 -0.0594 [-0.0839, -0.0079], BA cost 0.00295 [0.00097, 0.00681]; audit EOD6 is unchanged because its FPR extrema lie on G6 groups outside G3. RQ4a-6 Test paired deltas: XGBoost delta EOD6 -0.0867 [-0.1574, 0.0915], BA cost 0.00307 [0.00108, 0.01388]; RealMLP delta EOD6 -0.0656 [-0.2213, 0.1446], BA cost 0.00512 [0.00209, 0.03150]. C6 technical verification of RQ4a-3 is `PASS`; C7 technical verification of RQ4a-6 is `PASS`; RQ4a-3 artifacts kept identical SHA-256 and modification times through the RQ4a-6 run. C8 final review is `PASS`. Notebook 11: JSON and all 14 code cells validate, execution counts are sequential `1–14`, no saved error is present, reporting-file hashes match each run's artifact index, the saved key values match the reporting artifacts exactly, and the notebook imports no Fairlearn and reads no raw data, predictions, decision files, or bootstrap indices. | The C6 extrema interpretation is resolved: stored `*_extremum_role` values span all reported race groups including coverage-only groups, whereas EOD uses only the fixed roster; the RealMLP G6 FPR minimum is AIAN. The RQ4a-6 invalid-replicate path was not exercised in production (no G6 TPR/FPR gate failure in the locked Test rows) and is evidenced only by synthetic tests. Notebook 11 executed through the Anaconda `nbconvert` launcher with kernel `python3`, which resolved to Python 3.13.9 (pandas 2.3.3, matplotlib 3.10.6), not the RQ4a production environment. pytest is not part of the locked RQ4a environment. RQ4 outputs are exploratory model-behaviour evidence and do not establish legal compliance, discrimination, causal effects, creditworthiness, or default risk. RQ4b-3 reports its Validation reuse optimism limitation (shared threshold calibration and lambda selection). It has only three seeds for lambdas 0/0.01 and single-seed screening arms, the `training-unstable beyond selected epoch` label for lambda > 0, `stability_status: not_assessed`, and a real capability probe that is not fitted (`capability_inconclusive`). Its architecture, seed, refit horizon, full probe optimizer/stopping, stored-score log-ratio, fixed synthetic fixture, linear percentile CI/validity rules and strong-support endpoints are user-locked in the RQ4 plan/YAML. | RQ4a and the manuscript-equivalent RQ3a are closed after C8 `PASS`. Preserve both RQ4a run roots and notebook 11 unchanged. RQ4b-3 is closed: preserve its three source runs, the Gate C run, the Test run and Notebook 12 unchanged, and do not rerun or regenerate. The user records the RQ4b-3 research interpretation (Notebook 12 section 14). Codex post-handover re-check of RQ4b-3 Tasks 4–7 is pending. RQ4b-6 requires a separate new-stream authorization. | 2026-10-24 |
| RQ4b production contract and input boundary (T1-R1) | COMPLETE | Codex C1 re-review PASS on 2026-10-18; limited to T1 input-boundary implementation and synthetic validation. | `src/dataset_b/mitigation/rq4b_inputs.py`; `tests/test_dataset_b_rq4b_inputs.py`; `scratch/rq4b_implementation/T1_R1_C1_e8bfd4d9b56b4d1c8326026950918f11/`; implementer audit `scratch/rq4b_implementation/T1_R1_audit_b2b8f2858aad422e817e9b031f97c9f8/` | Codex reran focused pytest: 185 passed in 6.71s, exit 0; 704 independent locked-leaf mutation/deletion checks rejected invalid contracts; 27/27 outside-scope baseline hashes matched; both target hashes matched the implementer report; git diff --check found no whitespace errors. Both original C1 findings are closed: execution-critical contract coverage and authorized loader-resource binding. | No real split or frozen-prediction body reads, model fitting, inference, CUDA validation or scientific capability assessment. Successful Test-partition loading was not exercised. Contract objects remain caller-controlled; test docstring still understates YAML reads. Review command output is retained in the chat; the scratch root contains synthetic fixtures, not a saved Codex review log. | T1 accepted only. T2 requires separate assignment; remaining production implementation/validation and explicit user training launch remain required. RQ4b-6 stays deferred. | 2026-10-18 |
| RQ4b adversarial training and CV/refit engine (T2-R1) | COMPLETE | Codex C2 re-review PASS on 2026-10-18; T2 engine implementation accepted only within synthetic technical validation scope. | `src/dataset_b/mitigation/rq4b_training.py`; `tests/test_dataset_b_rq4b_training.py`; Codex scratch `scratch/rq4b_implementation/T2_R1_C2_7c87d0f8b72945aebe5aed34c2afd678/`; implementer audit `scratch/rq4b_implementation/T2_R1_audit_8f5a028f76f44e5db36087b760b65340/` | Codex reran T2 and T1 suites together: 267 passed (82 T2 plus 185 T1), 28 deprecation warnings, 38.05s, exit 0. Independent reproduction of the prior C2 finding now rejects foreign population IDs before preprocessing and engine calls; matching membership passes. Both target SHA-256 values match the implementer report; 32/32 outside-scope baseline files match; git diff --check found no whitespace errors. | GPU evidence is synthetic smoke of at most two epochs per fit. No production fit, 256-epoch execution or real Train/Validation/Test access. Full 5x256 bookkeeping is test-double evidence only. Membership digest covers fold-validation IDs under the locked label-stratified policy, not feature values; population loading and derived-race join integration remain outside T2. No scientific capability or production-readiness conclusion. Review command output is retained in the chat; the Codex scratch root contains test outputs, not a saved review transcript. | Prior C2 lineage finding closed. T3 requires separate assignment; remaining production implementation/validation and explicit user launch remain required. RQ4b-6 remains deferred. | 2026-10-18 |
| RQ4b capability probe (T3-R2) | COMPLETE | Codex C3 re-review PASS on 2026-10-18 for probe implementation, retained synthetic correctness gate and Windows attempt failure handling. Real-data capability remains unassessed. | `src/dataset_b/mitigation/rq4b_probe.py`; `tests/test_dataset_b_rq4b_probe.py`; `scratch/rq4b_implementation/T3_R2_C3_4480ba14a8c245e2923d91cf7faa64f3/`; retained `scratch/rq4b_implementation/T3_synthetic_gate_5c041ca3b32e481f89574148ee0cbf7f/synthetic_gate_evidence.json`; archived gate-producing source `scratch/rq4b_implementation/T3_R1_audit_f6913bbaca814a31a506af0368f79e50/rq4b_probe_gate_lineage_02ED1280.py.txt` | Codex reran the probe suite: 53 passed in 10.84s, exit 0. Independent injected failures confirm partial evidence is not published and a receipt is retained; receipt-write failure preserves the original exception identity and exposes the secondary error. Source/test hashes match the report; 39/39 outside-scope baseline files match, including retained gate and source lineage. Method functions remain unchanged since the gate; changed experiment/writer wrappers were reviewed. Retained gate: gain 1.0934503013695 nats, linear percentile 95% CI [1.0926187243623788, 1.0941724543148197], 1000/1000 valid replicates; best epoch 100, stopped at cap. Stored interval arithmetic independently matches. | Gate was not rerun; its producing code hash is 02ED1280C365FA23AAC71EDBC3799D7497B187EFE5B99CA2185DA94CA3F08C1D, distinct from current wrapper code. No real data/frozen prediction-body access or real-probe fitting. Real-route tests use synthetic data. No-overwrite publication is accepted on Windows; POSIX race safety is not established. Invocation-writing failures occur before fitting and may leave no receipt; receipt-writing failure cannot guarantee a receipt, but preserves the primary exception and secondary diagnostic. T1/T2 regression suites were not rerun in this re-review. Review outputs are retained in the chat, not a saved review log. | C3 findings closed. Real frozen-Validation probe fitting and the next implementation task require separate authorization. No production training authorization; RQ4b-6 remains deferred. | 2026-10-18 |
| RQ4b candidate selection and Test evaluation logic (T4-R1) | COMPLETE | Codex C4 re-review PASS on 2026-10-18; pure logic accepted on synthetic fixtures only. | `src/dataset_b/mitigation/rq4b_evaluation.py`; `tests/test_dataset_b_rq4b_evaluation.py`; `scratch/rq4b_implementation/T4_R1_C4_c61f2ec04ca947a7bb72c690a5a0dade/`; implementer audit `scratch/rq4b_implementation/T4_R1_audit_65de53ee4b294e98912c41fb9a705427/` | Codex reran 233 tests together (38 T4, 185 T1, 10 RQ4a metrics): all passed in 47.31s, exit 0. Prior C4 raw-sex-code reproduction now rejects invalid encoding at construction; explicit normalization maps all five approved RQ3 codes. Coverage-only labels remain valid and counted; genuine absent/insufficient groups retain undefined-metric semantics. Both target hashes match; 42/42 outside-scope baseline files match; targeted whitespace checks pass. AST comparison confirms existing changes limited to prediction validation and arm audit output, plus three sex helpers. Earlier C4 independent synthetic recomputation of 1000 paired replicates matched delta BA/EOD3/EOD6 values and CIs exactly. | No real Validation/Test data or locked bootstrap file access, model fitting or scientific outcome. Evaluation expects canonical sex labels; future loaders must explicitly normalize split codes. Locked-index SHA integrity, Test record-ID/manifest alignment, exact production row counts and enforcement of development-artifact reuse remain loader/integration gates. The sex mapping is cross-checked when normalization is called, not by reading YAML at prediction construction. Review output is retained in chat; scratch contains test fixtures, not a saved review log. | C4 sex-metadata finding closed. Next implementation task requires separate assignment; real probe and production training require separate authorization. RQ4b-6 remains deferred. | 2026-10-18 |
| RQ4b staged runner, reporting and package validation (T5-R1) | COMPLETE | Codex C5 re-review PASS on 2026-10-18, limited to fake-backend synthetic integration. T1-T5 implementation gates are accepted in their documented scopes; no production run is implied. | `src/dataset_b/mitigation/rq4b_runner.py`; `tests/test_dataset_b_rq4b_runner.py`; implementer audit `scratch/rq4b_implementation/T5_R1_audit_703729e717d641878fe067adad1535ea/`; independent synthetic reproduction `scratch/rq4b_implementation/T5_R1_C5_trace_eb1daf9865194922ab417a441fa4a7ab/` | Codex reran runner suite: 42 passed in 198.89s; T1 plus T4 regression: 223 passed in 13.19s, both exit 0. Independent EOD3/EOD6 generalization-gap calculation matches exactly at locked thresholds. Prior C5 forgery reproduction now rejects modified lineage despite internally consistent output hashes. Tests cover status, lineage, gap and per-artifact Test-ID forgeries plus intact packages. Both target hashes match; 44/44 outside-scope baseline files match before this acceptance-record update; targeted whitespace checks pass. Runner exports and revalidates gap from saved Validation/Test predictions without additional inference. | ProductionBackend, real-data preflight, production index loading and CLI dispatch have not been runtime-verified. All integration results and full-grid fit counts are fake-backend evidence, not observed production fits. Production counts are per-unit declarations checked against engine/curve contracts. No production namespace exists at review time; no real data, training or inference performed. Some failed stages have no retry flag and consume the run ID; repeated run IDs are unique only within each run. Real probe remains not fitted/capability_inconclusive. Directory publication relies on Windows semantics. Review output is retained in chat, not a saved review log. | Both C5 findings closed. C5 recording satisfies the runner acceptance prerequisite only; it does not launch or authorize production. Prepare exact staged commands and obtain explicit user launch before real execution; do not run the whole grid automatically. RQ4b-6 remains deferred. | 2026-10-18 |
| RQ4b internal-extension technical spike (S1-R1) | COMPLETE | Codex C-spike review PASS, limited to synthetic fixtures and at most two epochs per fit; not production readiness or a production result. | `src/dataset_b/mitigation/rq4b_spike.py`; `tests/test_dataset_b_rq4b_spike.py`; `scratch/rq4b_spike/r1_evidence_20261018_a1/`; `scratch/rq4b_spike/r1_pytest_20261018_a1/output.txt` | Stored implementer pytest log: 38 passed, exit 0. Codex independently reproduced installed-source comparison (89 files against saved upstream evidence), signed gradients/device/updates, lambda=0 equivalence under locked tolerances, exact in-memory reload for lambda=0 and positive lambda, row-feature-group alignment and same-label corruption rejection, nonfinite rejection before backward/step, mandatory non-PASS rejection, epoch selector and tolerance-boundary checks. R1 baseline hash audit: 77/77 outside the two edited files matched. | Codex did not rerun the full 38-test suite or filesystem receipt suite, fetch upstream evidence anew, or execute production CV; no full-run determinism claim. g10 is NOT_RUN in the evidence runner and covered by retained tests and focused review. Schedule feasibility is short-smoke evidence only. | Preserve spike evidence. Final probe and numerical decisions are locked; separately implement/validate production, and only the user launches training. The new probe protocol and CI rules are documented decisions, not executed results. | 2026-10-18 |
| RMLP-07C related test-suite result | COMPLETE | The current nine-module RealMLP suite passed in the RealMLP GPU environment; this supersedes the uncorroborated historical claim of 42 passed tests. | `artifacts/dataset_b/realmlp_v1_recovery_20260928/review/test_report.json`; related source tests under `tests/test_dataset_b_realmlp*.py` | Exact retained invocation reports 52 passed, 0 failed, 0 errors, and one non-blocking Torch JIT deprecation warning in 28.12 seconds. | None for the retained test run. This test result does not expand the experimental feature-lock scope. | Preserve the RMLP-10 review package with the RealMLP evidence. | 2026-09-28 |
| Documentation status reconciliation | COMPLETE | The execution register, machine-readable protocols, contracts, production evidence, visualization notebooks, and Dataset B technical DOCX are aligned on the accepted RQ2 and RQ3 closures. Historical GPU training/freeze evidence remains distinct from CPU-only RQ2 explanation inference; RQ3 remains an artifact-only descriptive audit. | This file; `contracts/{rq2_explanation,rq3_fairness}.md`; `config/dataset_b/{rq2_protocol,rq3_protocol}.yaml`; accepted RQ2/RQ3 artifact directories; `notebooks/dataset_b/{08_rq2_gsv_sl_results,09_rq3_fairness_results}.ipynb`; `../Yêu cầu Kỹ thuật Set B cho RQ1 và RQ2 và RQ3 - HMDA NY 2024 - Revised.docx` | RQ2 sources agree on run `rq2_cpu_production_20261009_01`, six-model PASS, 60 cross-model rows, observed runtimes, and no Test use. RQ3 sources agree on run `rq3_fairness_20261010_01`, six-model PASS, 43,996 Test rows, Approved-positive metrics, artifact-only notebook execution, and no fit, prediction, threshold, or bootstrap-index regeneration. | RQ2 agreement/stability and RQ3 group differences remain descriptive; neither establishes causal validity, legal discrimination, creditworthiness, default risk, or general explanation correctness. | Preserve both accepted runs, their interpretation boundaries, and their read-only notebook consumers. | 2026-10-10 |

## Status rules

- `COMPLETE` applies only to the named stage and stated evidence scope; it does not imply final research completion.
- `PARTIAL` means some named stages have evidence but required downstream work remains.
- `PENDING` means work has not been established as started/completed, or a required review remains open.
- `BLOCKED` means an explicit prerequisite or decision gate is unmet.
- `UNVERIFIED` is reserved for a claim lacking sufficient evidence; it is not a synonym for failure.
- `PLANNED / NOT IMPLEMENTED` means governance or configuration is recorded, but no implementation or execution evidence is claimed.
