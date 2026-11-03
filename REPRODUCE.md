# Dataset B Reproducibility Guide

This guide is the documentation-level reproducibility entry point for Dataset B.
It distinguishes current static verification, accepted historical runtime
evidence, procedures that were not rerun, and limitations that remain
unverified. It does not claim a clean installation, fresh training, checkpoint
evaluation, inference, notebook execution, or GPU execution in the current
documentation task.

The authoritative documentation acceptance register is
[REPRODUCIBILITY_ACCEPTANCE](docs/handbook/REPRODUCIBILITY_ACCEPTANCE.md).
Experimental execution status remains exclusively in
[STATUS](docs/handbook/STATUS.md).

## 0. General information

| Field | Evidence-backed value |
|---|---|
| Repository | `https://github.com/DAP391m-Group12/DAP391m.git` |
| Repository title | `DAP391m Group 12` |
| Research scope | Mortgage-application approval prediction for the documented 2024 HMDA New York Dataset B cohort |
| Course code | `DAP391m` |
| Vietnamese/English formal project titles | `NOT_DECLARED` in the inspected repository evidence |
| Semester, supervisor, member roster, student IDs, and project contact list | `NOT_DECLARED` |
| Submission identity | PR #12 merge commit `369fe494506fce888d6c917f504d086a1dfcb097` |
| Reviewed PR head | `ed1afbf783db502ade5dfb959738df51cdb6a5e7` |
| Document version | `2.1` |
| Document date | `2026-11-02` |

Three identities must not be conflated:

- **Submission identity** is the Git commit used to inspect repository source
  and documentation. For this guide it is the PR #12 merge commit above.
- **Accepted run identity** is a run-scoped identifier stored in manifests,
  receipts, thresholds, predictions, and reports. The six frozen RQ1 identities
  are `lr_final_15f_20261002_01`, `rf_final_15f_20261002_01`,
  `xgb_final_15f_20261003_01`, `plain_mlp_final_15f_20261004_03`,
  `realmlp_final_15f_20261004_01`, and
  `tabmmini_final_15f_20261004_01`.
- **Accepted historical evidence** is retained evidence produced before this
  documentation task. Its accepted state is recorded by
  [STATUS](docs/handbook/STATUS.md); it is not a claim that the task reran it.

The local retained data, artifacts, and checkpoints are not tracked by Git.
No public download location for those retained files is declared in the
repository. A clean clone therefore provides source and documentation, but not
the retained-artifact verification route below.

## 1. Quick Start

Run every command from the repository root. These routes are read-only. They do
not load a model or checkpoint, run inference, fit a model, execute a notebook,
or regenerate a report.

For this documentation acceptance, **Quick Start success** means that a reviewer
can navigate the repository, validate its configuration and saved notebook
structure, enumerate durable reports, and verify retained frozen locks when the
local artifacts are present. It does not mean that the reviewer recalculates a
headline metric. Numerical results remain accepted historical evidence tied to
the run identities, manifests, and reports in Sections 7--10.

### 1.1 Repository and documentation verification

This route requires an already available compatible Python environment. No
clean-host installation was attempted or accepted in this task.

```powershell
git clone https://github.com/DAP391m-Group12/DAP391m.git
Set-Location DAP391m
git checkout 369fe494506fce888d6c917f504d086a1dfcb097
python -B -m src.dataset_b.cli validate-config
python -B -m src.dataset_b.cli validate-notebooks --scope final
python -B -m src.dataset_b.cli report-index
```

Expected static results:

- `validate-config`: `status = PASS`, `dataset = dataset_b`, and
  `target_count = 11`;
- `validate-notebooks`: `status = PASS`, `notebook_count = 14`,
  `saved_error_count = 0`, and `executed = false`;
- `report-index`: `status = PASS`, `notebook_count = 14`, and the durable
  report paths that exist in that checkout.

The notebook result means saved JSON was parsed. It does not mean a kernel was
started or that `Restart & Run All` passed.

### 1.2 Retained-artifact verification

Use this route only in an authorized workspace that already contains the
retained Dataset B artifacts at their repository-relative paths.

```powershell
python -B -m src.dataset_b.cli status
python -B -m src.dataset_b.cli validate-artifacts --scope frozen
```

Expected retained-workspace results:

- `status`: `status = PASS`, `active_dataset = dataset_b`, six model markers
  are present, and training/inference/notebook-execution safety flags are
  `false`;
- `validate-artifacts`: `status = PASS`, `frozen_marker_count = 6`, three root
  locks match, `checkpoints_loaded = 0`, and `inference_calls = 0`.

On a clean clone without retained artifacts, `status` may be `INCOMPLETE` and
`validate-artifacts` must fail on a missing locked file. That is the documented
distribution limitation, not a successful retained-artifact check.

## 2. Repository structure

```text
config/                  project and Dataset B configuration contracts
data/raw/                immutable HMDA source location; not Git-distributed
data/dataset_b/          derived Dataset B split material; not Git-distributed
artifacts/dataset_b/     retained frozen and research evidence; not Git-distributed
notebooks/dataset_b/     historical and final reporting notebooks
src/dataset_b/           Dataset B implementation and read-only CLI
docs/handbook/           contracts, runbooks, STATUS, and acceptance register
reports/                 durable metrics and publication outputs
tests/                   static, contract, and implementation tests
```

Use these routers:

- [project README](README.md) for the Dataset A/Dataset B boundary;
- [handbook router](docs/handbook/README.md) for contracts and runbooks;
- [configuration router](config/README.md) for logical configuration roles;
- [notebook router](notebooks/dataset_b/README.md) for notebook roles;
- [source router](src/dataset_b/README.md) for canonical source ownership;
- [STATUS](docs/handbook/STATUS.md) for accepted execution state only.

## 3. Hardware and software

### 3.1 Accepted actual environments

The table reports only fields present in accepted manifests or STATUS. A field
marked `NOT_RECORDED` must not be filled from the current workstation.

| Evidence scope | OS | Python | Compute | Packages recorded | Runtime evidence |
|---|---|---|---|---|---|
| Classical RQ1 manifests | Windows-family is inferred from retained paths and the CPython MSC build; exact edition `NOT_RECORDED` | 3.13.9 | CPU identity and RAM `NOT_RECORDED` | NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, joblib 1.5.2; XGBoost 3.3.0 for its accepted run | LR full fit 2.85 s; RF full fit 465.21 s; XGBoost full fit 2.52 s |
| PlainMLP accepted run | OS, CPU, and RAM `NOT_RECORDED` | `NOT_RECORDED` in the inspected final manifest | NVIDIA GeForce RTX 5050 Laptop GPU; 8,546,484,224 bytes device memory; CUDA build 13.0 | Exact final training package set `NOT_RECORDED` in the inspected manifest | selector 107.37 s and final fit 95.46 s |
| RealMLP accepted run | Windows-family path; exact edition, CPU, and RAM `NOT_RECORDED` | 3.13.14 | NVIDIA GeForce RTX 5050 Laptop GPU, `cuda:0`; VRAM and driver `NOT_RECORDED` in the RealMLP runtime manifest | Torch 2.14.0+cu130, PyTabKit 1.7.3, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.1 | runtime smoke 1.94 s; full fit 639.62 s; Test inference 1.58 s |
| TabMmini accepted run | OS, CPU, and RAM `NOT_RECORDED` | `NOT_RECORDED` | NVIDIA GeForce RTX 5050 Laptop GPU, `cuda:0`; 8,546,484,224 bytes device memory; CUDA build 13.0 | Torch 2.14.0+cu130, TabM 0.0.3 | runtime smoke 1.52 s; full-Train runtime 377.69 s |
| RQ4a CPU environment | Windows-family MSC build; exact edition, CPU, and RAM `NOT_RECORDED` | 3.13.14 | CPU; CUDA not checked | NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, PyYAML 6.0.3, joblib 1.6.0, Fairlearn 0.15.0 | accepted environment smoke; production evidence is separately recorded in STATUS |

NVIDIA driver and cuDNN versions are `NOT_RECORDED` in the inspected accepted
manifests. CPU model, host RAM, and peak VRAM are also not established for all
families. The RQ2 accepted CPU run records 4,817.42 seconds total across six
models; RQ3 records per-model totals from 87.52 to 175.75 seconds.

The accepted neural evidence identifies an NVIDIA GeForce RTX 5050 Laptop GPU.
PlainMLP and TabMmini record 8,546,484,224 bytes of device memory (nominal 8 GB)
and CUDA build 13.0. The RealMLP manifest identifies the same GPU model and
`cuda:0`, but does not independently record VRAM. Documentation acceptance does
not require a new CUDA smoke run.

Package versions must be reported exactly as stored by each accepted run. If
successful retained evidence exists for more than one version of a package,
the lowest observed successful version may be labeled **lowest verified working
version**. This label is evidence-bounded; it is not a general minimum-supported
version. When only one successful version is retained, report that exact version
and do not infer a lower compatibility floor.

### 3.2 Recommended inspection environment

[`pyproject.toml`](pyproject.toml) requires Python 3.10 or newer.
[`requirements.txt`](requirements.txt) is the current pinned, CPU-oriented
dependency declaration. It is a declaration, not accepted proof that a fresh
resolver/install succeeds on every host. Neural CUDA routes require their
model-specific runbooks and compatible hardware; never silently fall back to
CPU where the contract forbids it.

## 4. Environment installation

The repository currently declares one installation route:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Classification: `DOCUMENTED_NOT_RERUN`. The command was not executed during
this task, and clean-install success is not claimed. The file pins current
package versions, but the accepted historical runs used more than one runtime
profile; consult the relevant manifest before attempting numerical reproduction.
Differences between profiles are recorded rather than normalized. A lowest
verified working version may be stated only under the evidence rule in Section
3.1; no package downgrade, reinstall, or CUDA smoke is required for this
documentation acceptance.

No conda lockfile, Dockerfile, GPU wheel lock, or accepted clean-host walkthrough
is present. Do not modify the pinned environment merely to force a match with a
historical result. Record the interpreter, platform, resolver error, CUDA build,
driver, and GPU identity first.

## 5. Data artifacts and checkpoints

### 5.1 Data source and split

Dataset B uses the 2024 HMDA loan-level data for New York. The project source
route is the [official HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/maps/2024).
The repository does not declare a separate data license identifier; terms and
license status are therefore `NOT_DECLARED` here.

| Item | Role | Local path | Size (bytes) | SHA-256 | Distribution |
|---|---|---|---:|---|---|
| HMDA NY 2024 CSV | immutable raw source | `data/raw/state_NY_2024.csv` | 145,740,132 | `e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d` | not tracked by Git; no repository download handoff |
| split v1 manifest | ordered Train/Validation/Test membership | `artifacts/dataset_b/split_v1.json` | 8,212,697 | `65d15274095076dbd2306ac7c7b25cf02f1032ed4b2410b30e1d00e3f5b446ee` | retained local evidence only |
| shared Test bootstrap indices | 1,000 locked paired-bootstrap replicates | `artifacts/dataset_b/rq1_test_bootstrap_indices.npz` | 113,753,708 | `2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1` | retained local evidence only |
| bootstrap manifest | index contract and lineage | `artifacts/dataset_b/rq1_test_bootstrap_manifest.json` | 917 | `7b202b9523820c4ec6b2e777adf1a672a43c6bd39ae42842fd52726d1e88a1cb` | retained local evidence only |

Verify one retained file without loading it:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath artifacts/dataset_b/split_v1.json
```

The split is 205,310 Train, 43,995 Validation, and 43,996 Test records, using a
70/15/15 stratified split with seed 42. The authoritative cohort, target,
exclusions, audit roles, and preprocessing boundary are in the
[data contract](docs/handbook/contracts/data_contract.md) and
[split/evaluation contract](docs/handbook/contracts/split_and_evaluation.md).

### 5.2 Frozen checkpoint inventory

These are retained local artifacts, not clean-clone content. The listed digest
is either manifest-pinned or was read directly from the retained frozen file
during documentation acceptance; neither operation loaded a checkpoint.

| Model/run | Primary retained checkpoint | Size (bytes) | SHA-256 | Classification |
|---|---|---:|---|---|
| LR `lr_final_15f_20261002_01` | `artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib` | 10,099 | `69770bc11558f56f147e2af89f79747c903e74e779aeae17d1347a25de620859` | accepted historical evidence |
| RF `rf_final_15f_20261002_01` | `artifacts/dataset_b/rf/final/rf_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib` | 596,820,036 | `2e2378596ba62f4c82b7d657d45cb733223612fb38943a5e121a818183b6c802` | accepted historical evidence |
| XGBoost `xgb_final_15f_20261003_01` | `artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01/checkpoints/fitted_pipeline.joblib` | 400,438 | `d24db9bb642e6e7810596e4a16632b1be88e68eeb030ad05130b134ea65ee834` | accepted historical evidence |
| PlainMLP `plain_mlp_final_15f_20261004_03` | `artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03/full_train/checkpoints/fitted_bundle.joblib` | 446,219 | `bad3761eeb72f41b6cb2784530a1f46fce6b9bb2948c812f7d6e5e4337ae1968` | accepted historical evidence |
| RealMLP `realmlp_final_15f_20261004_01` | `artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/full_train/fitted_model.joblib` | 663,830 | `b5b79dcd3200f989ecd9cdae850fccaeb2f98ae0487046c68cebbdbcf8ea2754` | accepted historical evidence |
| TabMmini `tabmmini_final_15f_20261004_01` | `artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01/full_train/model/inference_bundle.pt` | 1,272,134 | `0d3eacadb7a2e07ae5a6fb7396f87b00317a21711a0b0db8b6aaf4ec1e4d2229` | accepted historical evidence |

## 6. Configuration and hyperparameters

Run `python -B -m src.dataset_b.cli validate-config` to validate the eleven
logical targets. The [Dataset B catalog](config/dataset_b/catalog.yaml) is a
router; scientific values remain in the target YAML files.

| Family | Authoritative config | Locked highlights | Search status |
|---|---|---|---|
| LR | `config/dataset_b/model_protocol_v1.yaml` | L2, `C=1`, lbfgs, max_iter 1000, balanced class weight | fixed; no search |
| RF | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, gini, sqrt features, balanced class weight, seed 42 | fixed; no search |
| XGBoost | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, depth 6, learning rate 0.3, hist, seed 42 | fixed; no search |
| PlainMLP | `config/dataset_b/plain_mlp_v1.yaml` | 256/128/64 ReLU, AdamW 0.001, dropout 0.1, deterministic algorithms, seed 42 | locked configuration; no search |
| RealMLP | `config/dataset_b/realmlp_v1.yaml` | three 256-unit layers, SELU, Adam, fixed epoch selected from Train-only pilot, seed 42 | locked configuration; no search |
| TabMmini | `config/dataset_b/tabmmini_v1.yaml` | k=32, two 512 blocks, AdamW 0.002, fixed epoch from Train-only pilot, seed 42 | locked configuration; no search |

RQ2, RQ3, RQ4a, and RQ4b use their respective YAML files routed by the catalog.
The accepted results must be read with the exact config and run identity stored
in their manifests; current YAML status fields do not supersede STATUS when a
frozen config was intentionally left unchanged for lineage.

## 7. Training

No training was run in this documentation task. This section first records only
training that accepted historical evidence says was completed; it then separates
unrerun procedures so that a runbook is never presented as runtime evidence.

### 7.1 Accepted completed-run inventory

| Scope | Accepted completed runs | Recorded seed scope | Evidence-backed completion |
|---|---|---|---|
| Classical RQ1 | `lr_final_15f_20261002_01`, `rf_final_15f_20261002_01`, `xgb_final_15f_20261003_01` | split/CV seed 42; RF/XGBoost random state 42 | five fresh Train-only CV fits and a fresh full-Train refit per family; frozen checkpoint and Test package retained |
| PlainMLP RQ1 | `plain_mlp_final_15f_20261004_03` | seed 42; deterministic-algorithm policy recorded | Train-only selection/CV and fresh full-Train fit retained as accepted historical evidence |
| RealMLP RQ1 | `realmlp_final_15f_20261004_01` | seed 42; fixed epoch 50 selected from the Train-only pilot | five fresh Train-only CV fits and a fresh full-Train fit retained as accepted historical evidence |
| TabMmini RQ1 | `tabmmini_final_15f_20261004_01` | pilot/CV seed 42; fixed accepted full-Train epoch 55 | Train-only pilot/CV and fresh full-Train fit retained as accepted historical evidence |
| RQ4b-3 development | `rq4b3_adv_20261019_02`, `rq4b3_adv_20261021_01`, `rq4b3_adv_20261022_01` | lambdas 0 and 0.01 at seeds 42/43/44; screening lambdas 0.03/0.1/0.3 at seed 42 only | nine development units are recorded complete; Gate C selected lambda 0; closed evidence is not rerun |

This inventory does not require or authorize fresh training. It reports only
completed accepted runs and their actual seed coverage; it does not upgrade a
single-seed screening arm into multi-seed evidence.

### 7.2 Documented procedure routes

The following routes are procedures, not current runtime PASS claims.

| Family | Procedure | Classification |
|---|---|---|
| LR/RF/XGBoost | [baseline training runbook](docs/handbook/runbooks/baseline_training.md); regenerate run-scoped fold inputs with `python -m src.dataset_b.workflows.export_classical_cv --run-id $RUN_ID`, then use the model-specific internal implementation described by the runbook | `DOCUMENTED_NOT_RERUN`; there is no repository-level public training command |
| PlainMLP | [neural training runbook](docs/handbook/runbooks/neural_training.md) plus `plain_mlp_v1.yaml` | `ACCEPTED_HISTORICAL_EVIDENCE` for the frozen run; a new run is `DOCUMENTED_NOT_RERUN` and separately authorized |
| RealMLP | runbook procedures for pilot, CV, and full-Train refit; e.g. `python -m src.dataset_b.runtime.realmlp_cv run --run-id $RUN_ID` and `python -m src.dataset_b.runtime.realmlp_full_train run --run-id $RUN_ID` | `NOT_PUBLIC_HANDOVER_COMMAND`; operator/internal route, not executed here |
| TabMmini | [TabMmini training runbook](docs/handbook/runbooks/tabmmini_training.md) and locked config | `ACCEPTED_HISTORICAL_EVIDENCE` for the frozen run; no public handover training command |
| RQ4b adversarial | staged internal route in the neural runbook, one authorized unit at a time | `NOT_PUBLIC_HANDOVER_COMMAND`; closed evidence must not be rerun |

All families use five fresh Train-only CV fits and a fresh full-Train refit under
their owning contract. Validation selects the threshold; Test is final. A new
execution requires a new run ID and may not overwrite a frozen run.

## 8. Evaluation and inference

There is deliberately no public `evaluate`, `predict`, `inference`, or `serve`
subcommand in `src.dataset_b.cli`. The five public commands in Section 1 are
navigation and static integrity checks only.

Accepted historical evaluation evidence includes 43,996 ordered Test rows per
RQ1 model, frozen Validation thresholds, ten RQ1 metrics, and 1,000 locked
bootstrap replicates. RQ2 is CPU-only frozen inference over Validation and does
not use Test. RQ3 is an artifact-only Test audit with zero fit and zero
prediction calls. RQ4a/RQ4b execution state and limitations are recorded in
STATUS.

Model-facing evaluation procedures in runbooks are classified
`DOCUMENTED_NOT_RERUN` or `NOT_PUBLIC_HANDOVER_COMMAND`. This task did not load
any checkpoint, run inference, recalculate metrics from model outputs, or
execute a notebook.

## 9. Seeds and determinism

| Scope | Locked policy | Evidence/limitation |
|---|---|---|
| Split | seed 42; 70/15/15 stratified by `y_approved` | immutable `split_v1` |
| RQ1 CV | five-fold StratifiedKFold, shuffle, seed 42, Train only | accepted six-model run lineage |
| LR/RF/XGBoost | fixed configuration; RF/XGBoost random_state 42 | no hyperparameter search |
| PlainMLP | seed 42; deterministic algorithms true | GPU identity/version changes may still alter numerical behavior |
| RealMLP | seed 42; five fresh folds; fixed epoch 50 from Train-only pilot | deterministic cross-host identity is not claimed |
| TabMmini | seed 42; pilot and CV seed 42; fresh state | fixed epoch 55 in accepted full-Train result |
| RQ2 | sample/background/assignment seeds 4201/4202/4203; bootstrap seed 4204 | 1,000 shared replicates |
| RQ4b | model seeds 42, 43, 44 for lambdas 0 and 0.01; screening lambdas only seed 42 | three-seed summaries are not pooled with within-seed bootstrap rows |

Do not apply one tolerance to every model. PlainMLP compares replicate IDs and
validity payloads exactly, rounds `calibration_slope` to six decimals, uses
absolute tolerance `1e-14` for the other bootstrap metrics and CI endpoints,
and `1e-12` for point metrics. Other families retain their owning manifest,
review, and numerical policies. Exact SHA-256 is for stored-artifact integrity
and lineage, not a universal floating-point equivalence rule.

## 10. Result to report mapping

This mapping is inspectable without notebook execution.

| Reporting scope | Notebook/report | Accepted run/config | Procedure | Durable output | Classification |
|---|---|---|---|---|---|
| RQ1 six-model metrics | `notebooks/dataset_b/final/models/01_...06_*.ipynb`; `07_model_total_review.ipynb`; `reports/RQ1_Research_Metrics_Final.txt` | six frozen RQ1 run IDs; model configs in Section 6 | final notebook consumes frozen evidence | per-model reporting directories and RQ1 report | `ACCEPTED_HISTORICAL_EVIDENCE` |
| RQ2 explanations | `08_rq2_gsv_sl_results.ipynb`; `reports/RQ2_Research_Metrics_Final.txt` | `rq2_cpu_production_20261009_01`; `rq2_protocol.yaml` | CPU frozen inference; 256 explanation rows; 1,000 bootstrap replicates | `artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01/` | `ACCEPTED_HISTORICAL_EVIDENCE` |
| RQ3 fairness | `09_rq3_fairness_results.ipynb`; combined RQ3/RQ4 report | `rq3_fairness_20261010_01`; `rq3_protocol.yaml` | artifact-only audit; zero fit/prediction calls | `artifacts/dataset_b/rq3/rq3_fairness_20261010_01/` | `ACCEPTED_HISTORICAL_EVIDENCE` |
| RQ4a post-processing | `11_rq4a_equalized_odds_results.ipynb`; combined RQ3/RQ4 report | `rq4a3_postproc_20261017_01`, `rq4a6_postproc_20261017_01` | saved-prediction post-processing | run reporting directories | `ACCEPTED_HISTORICAL_EVIDENCE` with documented G6 limitations |
| RQ4b adversarial | `12_rq4b_adversarial_results.ipynb`; Figure 3 report output | three development runs, Gate C, and `rq4b3_test_20261024_01` | staged internal procedure; no rerun | Test/package reporting artifacts | `ACCEPTED_HISTORICAL_EVIDENCE`; interpretation remains separate |
| RQ4b sex supplement | `13_rq4b_applicant_sex_supplementary_results.ipynb` | `rq4b3_sex_supplementary_20261024_02` | evaluation from saved predictions | supplementary package | `ACCEPTED_HISTORICAL_EVIDENCE` |
| Publication figures and assembled report | `10_publication_figures.ipynb`, `14_research_report.ipynb`, `reports/paper_figures/` | inputs above | read/report procedure | PDFs and verification notes listed by `report-index` | `DOCUMENTED_NOT_RERUN` in this task |

The retained reports provide object-level provenance beneath this scope map:

- the RQ1 report records the six-model metric and bootstrap sources;
- the RQ2 report records its contract, config, accepted run, and notebook;
- the combined RQ3/RQ4 report includes Tables 2A, 2B, 5A, A1, and A2 with
  accepted run and source-artifact lineage;
- `reports/paper_figures/verification_notes.md` records the interpretation
  boundaries for Figures 2, 3, 5, and 6.

The scope-level map plus those retained report entries satisfies the current
documentation-acceptance mapping gate. It does not claim that every table or
figure was regenerated in this task. Per-object regeneration commands would be
needed only if the acceptance scope is expanded to a fresh report rebuild.

Notebook paths above are under `notebooks/dataset_b/final/`. The migration to
that directory was path-only; it did not execute or regenerate notebook output.

## 11. Limitations and troubleshooting

| Symptom or limitation | Meaning | Required action |
|---|---|---|
| Clean clone lacks raw data/artifacts/checkpoints | retained evidence is not Git-distributed and no public artifact handoff is declared | use repository/document verification only; do not claim retained-artifact PASS |
| Clean dependency installation fails | no clean-host install was accepted in this task | record OS, Python, resolver output, package index, GPU/driver/CUDA; do not edit frozen evidence |
| CUDA unavailable | CUDA-only route is not runnable on that host | stop; do not silently fall back to CPU |
| Raw CSV hash differs | it is not the exact historical project export | preserve the mismatch and do not relabel it as the frozen source |
| Frozen hash fails | stored lineage differs from the accepted lock | stop and investigate; do not regenerate or overwrite |
| Notebook contains saved error output | it is not clean saved reporting evidence | inspect the named notebook as JSON; do not execute or resave without separate authorization |
| Historical `D:\Proj\...` or `/content/drive/MyDrive/...` path appears | it is run-profile lineage, not a portable repository default | interpret against the recorded runtime; use repository-relative catalog paths for current navigation |

The frozen configuration and manifests intentionally preserve historical
absolute paths. Do not edit them for portability. Current navigation resolves
the project root from source and requires catalog targets to be
repository-relative. The Colab profile's `/content` and Google Drive roots are
profile-specific and its recorded runtime execution remains not started.

No clean installation, model execution, training, or GPU walkthrough was
performed in this documentation task. Cross-host determinism, driver/cuDNN
compatibility, RAM sufficiency, and end-to-end train-from-scratch remain
unverified.

## 12. License citation and AI declaration

License and citation completeness is excluded from the current documentation
acceptance score by owner decision. The `NOT_DECLARED` entries below are retained
as transparent inventory and are not acceptance failures. The generative-AI
declaration remains in scope.

| Item | Repository-backed status |
|---|---|
| Project code license | `NOT_DECLARED`; no LICENSE/COPYING file was found |
| HMDA data license/terms | official source URL is documented; a license identifier or copied terms are `NOT_DECLARED` |
| Project citation metadata | `NOT_DECLARED`; no CITATION file or project BibTeX is present |
| Model/library citations | package names and versions are recorded in configs/manifests; formal paper citations are `NOT_DECLARED` |
| Generative AI evidence | repository history and handbook record historical Claude implementation work and Codex implementation/review/documentation work; authorship and review roles must not be relabeled |

Users should cite the official HMDA source and the original libraries according
to their authoritative upstream guidance. This repository does not provide
enough metadata to invent a project BibTeX entry, select a code license, or
assert data/model terms on behalf of upstream owners.

The project predicts approval behavior for the documented cohort. It does not
predict default risk or creditworthiness, and its outputs are not causal, legal,
or policy evidence of discrimination.
