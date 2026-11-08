# Dataset B Reproducibility Guide

This guide is the reproducibility entry point for Dataset B. It distinguishes
static verification, runs observed on 2026-11-03 and 2026-11-08, accepted historical runtime
evidence, procedures that were not rerun, and limitations that remain unverified.
Section 1 gives the command route; the end-to-end clean-clone run has not been
executed, and no model was trained or fitted for this guide.

The qualified-host Phase B run on 2026-11-08 through 2026-11-09 passed asset
verification, all five environment profiles, six-model frozen-checkpoint evaluation,
all 402 RQ1 comparisons, and all four research routes. An initial RQ2 run failed three
TabMmini comparisons because the frozen CPU inference path did not lock Torch's
intra-op thread count. A one-thread lock scoped to each TabMmini inference call restored
the caller's setting afterwards and the official rerun passed all 28 RQ2 rows without
changing the numerical policy or accepted artifacts. `reports` then returned the
documented `INCOMPLETE` status: five objects passed and three notebook/authored-report
objects were not regenerated. Phase C was not opened in that task. See
`docs/handbook/STATUS.md` for the receipts and exact limitations.

Phase C subsequently rebuilt all 11 core ZIPs from the retained inventory. Every
bundle kept the size and SHA-256 declared before the build, every ZIP member matched
its assigned manifest inventory, RQ3 `runtime.csv` remained excluded, and the optional
RQ4 audit archive remained outside the core manifest. The verified local bundle total
is 678,068,350 bytes. The 11 core LFS pointers and objects were pushed on branch
`codex/cuda-phase-b-c-handoff` in release commit
`1d384482f7b00f84c06cdecbb4abd13380da91a2`. Public download URLs in the reproduction
manifest, anonymous download, and clean-clone installation remain unverified.

The authoritative documentation acceptance register is
[REPRODUCIBILITY_ACCEPTANCE](docs/handbook/REPRODUCIBILITY_ACCEPTANCE.md).
Experimental execution status remains exclusively in
[STATUS](docs/handbook/STATUS.md).

## 0. General information

| Field                               | Evidence-backed value                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| Repository                          | `https://github.com/DAP391m-Group12/DAP391m.git`                                              |
| Repository title                    | `DAP391m Group 12`                                                                            |
| Research scope                      | Mortgage-application approval prediction for the documented 2024 HMDA New York Dataset B cohort |
| Course code                         | `DAP391m`                                                                                     |
| Vietnamese project title            |                                                                                                 |
| English project title               |                                                                                                 |
| Semester                            |                                                                                                 |
| Supervisor                          |                                                                                                 |
| Members and roles                   |                                                                                                 |
| Student IDs                         |                                                                                                 |
| Contact                             |                                                                                                 |
| Submission identity                 | PR #12 merge commit `369fe494506fce888d6c917f504d086a1dfcb097`                                |
| Reviewed PR head                    | `ed1afbf783db502ade5dfb959738df51cdb6a5e7`                                                    |
| Reproduction route version          | commit `61a87bec7878c3507b0b76dd5fdae5d4d320b7ff` on `codex/cuda-phase-b-c-handoff`          |
| Final tag                           |                                                                                                 |
| Processed data and checkpoint links |                                                                                                 |
| Project bundle size                 | 11 bundles, 678,068,350 bytes in total (Section 5.5)                                            |
| Raw data link                       | `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=NY&years=2024`; 145,740,132 bytes |
| Document version                    | `2.4`                                                                                         |
| Document date                       | `2026-11-09`                                                                                  |

Three identities must not be conflated:

- **Submission identity** is the Git commit used to inspect repository source
  and documentation. The PR #12 merge commit above is the commit cited by the earlier
  documentation review. It contains no version-labeled guide, acceptance register, or
  reproduction CLI; those came in later commits. The owner's final tag replaces it at
  delivery.
- **Accepted run identity** is a run-scoped identifier stored in manifests,
  receipts, thresholds, predictions, and reports. The six frozen RQ1 identities
  are `lr_final_15f_20261002_01`, `rf_final_15f_20261002_01`,
  `xgb_final_15f_20261003_01`, `plain_mlp_final_15f_20261004_03`,
  `realmlp_final_15f_20261004_01`, and
  `tabmmini_final_15f_20261004_01`.
- **Accepted historical evidence** is retained evidence produced before this
  documentation task. Its accepted state is recorded by
  [STATUS](docs/handbook/STATUS.md); it is not a claim that the task reran it.

The retained data, artifacts, and checkpoints are not tracked by Git. A clean clone
provides source and documentation, then obtains them through Section 5.5.

## 1. Quick Start

Run every command from the repository root on Windows with PowerShell. Use Python 3.13
for the host; its lock needs Python 3.11 or newer, although `pyproject.toml` allows
3.10.

### 1.0 Reproduce the main result from frozen checkpoints

```powershell
git clone https://github.com/DAP391m-Group12/DAP391m.git
Set-Location DAP391m
git checkout 61a87bec7878c3507b0b76dd5fdae5d4d320b7ff
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install uv==0.11.22 -r environments/reproducibility/classical_cpu.txt
python -m src.dataset_b.reproduce environment --install   # about 12 minutes; about 11 GB
python -m src.dataset_b.reproduce download                # needs the links in Section 5.5
python -m src.dataset_b.reproduce verify-assets           # expected: 654 of 654 assets verified
python -m src.dataset_b.reproduce evaluate --model all    # about 6 minutes; no training or fitting
python -m src.dataset_b.reproduce compare                 # expected: 402 of 402 rows PASS
python -m src.dataset_b.reproduce reports                 # writes reports/reproducibility/reports/
```

Expected Test results, matching the RQ1 report (Section 10):

| Model               | ROC-AUC | Balanced Accuracy | Threshold |
| ------------------- | ------: | ----------------: | --------: |
| Logistic Regression |  0.8179 |            0.7384 |     0.493 |
| Random Forest       |  0.8409 |            0.7669 |     0.712 |
| XGBoost             |  0.8641 |            0.7830 |     0.525 |
| PlainMLP            |  0.8588 |            0.7778 |     0.543 |
| RealMLP             |  0.8632 |            0.7828 |     0.726 |
| TabMmini            |  0.8578 |            0.7799 |     0.508 |

Execution status of this route on 2026-11-03: `environment`, `verify-assets`,
`evaluate`, `compare`, and `reports` were each observed on the author machine from
a working tree whose assets came from verified local bundles. `evaluate` ran the
classical and neural models in separate invocations after a first full invocation
failed on a defect that was then fixed. `download` stops with exit code 3 until the
links are recorded. The bootstrap lines above and the complete sequence on a clean
machine were not executed; the observed commands used the author's Python 3.13.14 host
rather than the host lock. `reports` currently exits 1 with `INCOMPLETE`, because
Section 10 lists objects that were not run or regenerated. Parts of the CLI, including
`evaluate`, `compare`, and `reports`, were edited after those observations and are
covered since then by unit tests only; the commands were not repeated.

Current-source status on the qualified handoff host: `evaluate`, `compare`,
`research --route all`, and `reports` were repeated on 2026-11-08 through 2026-11-09.
Evaluation passed 6/6, RQ1 comparison passed 402/402, RQ2 passed 28/28, RQ3 passed
5/5, RQ4a passed 4/4, and RQ4b passed 17/17. `reports` remained `INCOMPLETE` only for
the three documented notebook/authored-report objects that have no source-backed
builder in the reproduction route. This was an in-place qualified-host run, not the
clean-clone route above.

Exit codes: 0 PASS; 1 FAIL or INCOMPLETE; 2 usage error; 3 gate BLOCKED. The stages
above and the RQ routes also run in order through
`python -m src.dataset_b.reproduce all`, or through `scripts/reproduce.ps1` followed
by the same arguments. The `all` route includes RQ2, whose accepted run took about
80 minutes.

The two routes below are read-only. They do not load a model or checkpoint, run
inference, fit a model, execute a notebook, or regenerate a report.

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
`validate-artifacts` must fail on a missing locked file.

## 2. Repository structure

```text
config/                  project and Dataset B configuration contracts
data/raw/                immutable HMDA source location; not Git-distributed
data/dataset_b/          derived Dataset B split material; not Git-distributed
artifacts/dataset_b/     retained frozen and research evidence; not Git-distributed
notebooks/dataset_b/     historical and final reporting notebooks
src/dataset_b/           Dataset B implementation and read-only CLI
src/dataset_b/reproduce/ reproduction CLI: assets, environments, evaluation, reports
config/dataset_b/reproducibility/  asset manifest, numerical policy, link template
environments/reproducibility/      exact package locks for each environment profile
scripts/reproduce.ps1    PowerShell wrapper for the reproduction CLI
docs/handbook/           contracts, runbooks, STATUS, and acceptance register
reports/                 durable metrics and publication outputs
reports/reproducibility/ generated reproduction outputs; never accepted runs
dist/reproducibility/    release bundles; Git LFS on the handoff branch
tests/                   static, contract, and implementation tests
```

The main logic is source-backed. Notebooks are numbered reporting consumers under
`notebooks/dataset_b/final/`.

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

| Evidence scope          | OS                                                                                                       | Python                                           | Compute                                                                                                          | Packages recorded                                                                                             | Runtime evidence                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Classical RQ1 manifests | Windows-family is inferred from retained paths and the CPython MSC build; exact edition `NOT_RECORDED` | 3.13.9                                           | CPU identity and RAM `NOT_RECORDED`                                                                            | NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, joblib 1.5.2; XGBoost 3.3.0 for its accepted run | LR full fit 2.85 s; RF full fit 465.21 s; XGBoost full fit 2.52 s                |
| PlainMLP accepted run   | OS, CPU, and RAM `NOT_RECORDED`                                                                        | `NOT_RECORDED` in the inspected final manifest | NVIDIA GeForce RTX 5050 Laptop GPU; 8,546,484,224 bytes device memory; CUDA build 13.0                           | Exact final training package set `NOT_RECORDED` in the inspected manifest                                   | selector 107.37 s and final fit 95.46 s                                          |
| RealMLP accepted run    | Windows-family path; exact edition, CPU, and RAM `NOT_RECORDED`                                        | 3.13.14                                          | NVIDIA GeForce RTX 5050 Laptop GPU, `cuda:0`; VRAM and driver `NOT_RECORDED` in the RealMLP runtime manifest | Torch 2.14.0+cu130, PyTabKit 1.7.3, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.1                             | runtime smoke 1.94 s; full fit 639.62 s; Test inference 1.58 s                   |
| TabMmini accepted run   | OS, CPU, and RAM `NOT_RECORDED`                                                                        | `NOT_RECORDED`                                 | NVIDIA GeForce RTX 5050 Laptop GPU, `cuda:0`; 8,546,484,224 bytes device memory; CUDA build 13.0               | Torch 2.14.0+cu130, TabM 0.0.3                                                                                | runtime smoke 1.52 s; full-Train runtime 377.69 s                                |
| RQ4a CPU environment    | Windows-family MSC build; exact edition, CPU, and RAM `NOT_RECORDED`                                   | 3.13.14                                          | CPU; CUDA not checked                                                                                            | NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, PyYAML 6.0.3, joblib 1.6.0, Fairlearn 0.15.0     | accepted environment smoke; production evidence is separately recorded in STATUS |

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

### 3.2 Configuration used for the 2026-11-03 route

| Item                             | Used                                                                                                   | Minimum recommended                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Operating system                 | Windows 11, build 10.0.26200                                                                           | no lower floor was tested; use the same family                                |
| GPU and VRAM                     | NVIDIA GeForce RTX 5050 Laptop GPU, 8,546,484,224 bytes; peak allocation 35 to 87 MB during evaluation | a CUDA GPU is required for the three neural models; no lower floor was tested |
| NVIDIA driver, CUDA build, cuDNN | 610.74; 13.0; 92400 as reported by torch                                                               | no lower floor was tested                                                     |
| Python                           | 3.13.9, 3.12.13, or 3.13.14, fixed per profile in Section 4                                            | exactly as locked                                                             |
| CPU and RAM                      | `NOT_RECORDED`                                                                                       | `NOT_RECORDED`                                                              |
| Disk                             | bundles 0.65 GB; placed assets 2.2 GB; environments about 11 GB                                        | about 14 GB                                                                   |
| Environment installation         | about 12 minutes on first download                                                                     | not applicable                                                                |
| Evaluation time                  | about 4 minutes for the classical models; about 40 seconds per neural model                            | not applicable                                                                |
| Training time                    | Section 3.1 records accepted fits; none exceeds 4 GPU hours, and none ran on 2026-11-03                | not applicable                                                                |

### 3.3 Recommended inspection environment

[`pyproject.toml`](pyproject.toml) requires Python 3.10 or newer.
[`requirements.txt`](requirements.txt) is the current pinned, CPU-oriented
dependency declaration. It is a declaration, not accepted proof that a fresh
resolver/install succeeds on every host. Neural CUDA routes require their
model-specific runbooks and compatible hardware; never silently fall back to
CPU where the contract forbids it.

## 4. Environment installation

Reproduction uses five exact locks under `environments/reproducibility/`. The first
step in Section 1.0 installs the command-line host; the command below creates the
profiles, which `evaluate`, `predict`, and `research` select automatically:

```powershell
python -m src.dataset_b.reproduce environment --install
```

| Profile            | Python  | Compute                  | Serves                                      | Lock file              |
| ------------------ | ------- | ------------------------ | ------------------------------------------- | ---------------------- |
| `classical_cpu`  | 3.13.9  | CPU                      | Logistic Regression, Random Forest, XGBoost | `classical_cpu.txt`  |
| `plain_mlp_cuda` | 3.12.13 | CUDA, torch 2.14.0+cu130 | PlainMLP                                    | `plain_mlp_cuda.txt` |
| `realmlp_cuda`   | 3.13.14 | CUDA, torch 2.14.0+cu130 | RealMLP                                     | `realmlp_cuda.txt`   |
| `tabmmini_cuda`  | 3.12.13 | CUDA, torch 2.14.0+cu130 | TabMmini                                    | `tabmmini_cuda.txt`  |
| `rq_cpu`         | 3.13.14 | CPU, torch 2.14.0+cpu    | RQ2, RQ3, RQ4a, RQ4b routes                 | `rq_cpu.txt`         |

`profiles.json` records each profile's smoke imports and where its versions come
from. The PlainMLP, TabMmini, and part of the `rq_cpu` versions are inferred from
matching environments, because those runs do not record a full package set. Profiles
were installed on the author machine on 2026-11-03.

To check an installation, run the following command. It exits 0 only when every
profile matches its lock, every smoke import works, and each CUDA profile completes
a finite CUDA tensor operation; it writes `reports/reproducibility/environment/`:

```powershell
python -m src.dataset_b.reproduce environment
```

The profile directory defaults to `.venv-repro/`; use `--env-root` for another location,
and pass the same option to `evaluate`, `predict`, `research`, and `all`.
[`requirements.txt`](requirements.txt) remains a general inspection environment and is
not used for numerical reproduction. No Dockerfile or conda lockfile is present.

No environment variable, API key, or secret is required by Dataset B source, so
the repository has no `.env.example`. `LOG_LEVEL` is an optional logging
override; PlainMLP sets `CUBLAS_WORKSPACE_CONFIG` itself when it is unset; and
the tracked configuration files contain no `${VAR}` placeholders.

## 5. Data artifacts and checkpoints

### 5.1 Data source and split

Dataset B uses the 2024 HMDA loan-level data for New York. The project source
route is the [official HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/maps/2024).
The [handbook](docs/handbook/README.md) records that the experimental file was
downloaded between 2026-09-07 and 2026-09-15, and that a 2026-10-01 download is kept for
comparison only. The hash below remains the identity check. Data terms are in Section 12.

| Item                          | Role                                     | Local path                                               | Size (bytes) | SHA-256                                                              | Distribution                             |
| ----------------------------- | ---------------------------------------- | -------------------------------------------------------- | -----------: | -------------------------------------------------------------------- | ---------------------------------------- |
| HMDA NY 2024 CSV              | immutable raw source                     | `data/raw/state_NY_2024.csv`                           |  145,740,132 | `e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d` | official source; not in a project bundle |
| split v1 manifest             | ordered Train/Validation/Test membership | `artifacts/dataset_b/split_v1.json`                    |    7,919,382 | `66816ba47da64bfd85c010a90a14b9d80443a46d319258bf9fc1b3657f731596` | project bundle                           |
| shared Test bootstrap indices | 1,000 locked paired-bootstrap replicates | `artifacts/dataset_b/rq1_test_bootstrap_indices.npz`   |  113,753,708 | `2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1` | project bundle                           |
| bootstrap manifest            | index contract and lineage               | `artifacts/dataset_b/rq1_test_bootstrap_manifest.json` |          886 | `5544af4ebe91bfdf8c5b3b574fa2ef24f3da8c14827f353970cbe9d759e9d21e` | project bundle                           |

Verify every declared asset without loading it:

```powershell
python -m src.dataset_b.reproduce verify-assets
```

To verify one file by hand:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath artifacts/dataset_b/split_v1.json
```

The split is 205,310 Train, 43,995 Validation, and 43,996 Test records, using a
70/15/15 stratified split with seed 42. The authoritative cohort, target,
exclusions, audit roles, and preprocessing boundary are in the
[data contract](docs/handbook/contracts/data_contract.md) and
[split/evaluation contract](docs/handbook/contracts/split_and_evaluation.md).

### 5.2 Frozen checkpoint inventory

These are not Git-distributed; Section 5.5 obtains them. The listed digest is
either manifest-pinned or was read directly from the retained frozen file.

| Model/run                                    | Primary retained checkpoint                                                                                         | Size (bytes) | SHA-256                                                              | Classification               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -----------: | -------------------------------------------------------------------- | ---------------------------- |
| LR `lr_final_15f_20261002_01`              | `artifacts/dataset_b/lr/final/lr_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib`                        |       10,099 | `69770bc11558f56f147e2af89f79747c903e74e779aeae17d1347a25de620859` | accepted historical evidence |
| RF `rf_final_15f_20261002_01`              | `artifacts/dataset_b/rf/final/rf_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib`                        |  596,820,036 | `2e2378596ba62f4c82b7d657d45cb733223612fb38943a5e121a818183b6c802` | accepted historical evidence |
| XGBoost `xgb_final_15f_20261003_01`        | `artifacts/dataset_b/xgboost/final/xgb_final_15f_20261003_01/checkpoints/fitted_pipeline.joblib`                  |      400,438 | `d24db9bb642e6e7810596e4a16632b1be88e68eeb030ad05130b134ea65ee834` | accepted historical evidence |
| PlainMLP `plain_mlp_final_15f_20261004_03` | `artifacts/dataset_b/plain_mlp/final/plain_mlp_final_15f_20261004_03/full_train/checkpoints/fitted_bundle.joblib` |      446,219 | `bad3761eeb72f41b6cb2784530a1f46fce6b9bb2948c812f7d6e5e4337ae1968` | accepted historical evidence |
| RealMLP `realmlp_final_15f_20261004_01`    | `artifacts/dataset_b/realmlp/final/realmlp_final_15f_20261004_01/full_train/fitted_model.joblib`                  |      663,830 | `b5b79dcd3200f989ecd9cdae850fccaeb2f98ae0487046c68cebbdbcf8ea2754` | accepted historical evidence |
| TabMmini `tabmmini_final_15f_20261004_01`  | `artifacts/dataset_b/tabmmini/final/tabmmini_final_15f_20261004_01/full_train/model/inference_bundle.pt`          |    1,272,134 | `0d3eacadb7a2e07ae5a6fb7396f87b00317a21711a0b0db8b6aaf4ec1e4d2229` | accepted historical evidence |

### 5.3 Data description

The counts below come from direct read-only counting of the hash-verified raw
file and split CSVs on 2026-11-03.

| Item                                                    | Value                                                                                                                                     |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Raw source                                              | 383,614 data rows and 99 columns                                                                                                          |
| Eligible cohort                                         | 293,301 rows with `action_taken` in `{1, 2, 3}`                                                                                       |
| Excluded rows                                           | 90,313 rows with other `action_taken` values (`4` 42,186; `5` 18,377; `6` 28,198; `7` 281; `8` 1,271)                         |
| Target                                                  | `y_approved = 1` for `action_taken` in `{1, 2}` (208,818 and 9,942 rows); `y_approved = 0` for `action_taken = 3` (74,541 rows) |
| Class balance                                           | 218,760 Approved (74.59%) and 74,541 Denied (25.41%)                                                                                      |
| Split CSV schema                                        | 27 columns: `record_id`, `action_taken`, `y_approved`, 15 model inputs, and 9 audit-only columns                                    |
| Blank or literal `NA` share among the 15 model inputs | `debt_to_income_ratio` 7.66%; `income` 7.02%; `loan_term` 3.53%; `loan_to_value_ratio` 3.40%; the other 11 inputs 0%              |

| Partition  |    Rows | Approved | Denied | Size (bytes) | SHA-256                                                              |
| ---------- | ------: | -------: | -----: | -----------: | -------------------------------------------------------------------- |
| Train      | 205,310 |  153,131 | 52,179 |   23,631,341 | `04575b2f68638bcfa3a9da3e61cdf30331986d8b5adf81034c4725cf8d692477` |
| Validation |  43,995 |   32,814 | 11,181 |    5,063,935 | `cd233414da1be43c4042b3c957994a1f222b48001b52e4057002129cf89d3cbb` |
| Test       |  43,996 |   32,815 | 11,181 |    5,064,953 | `5815caf720ea5a9b3a1c8925f9fe193062ce862ee6a78c534a3aa14b572cae05` |

Literal `NA` and `Exempt` are valid field states rather than imputed blanks; the
share above does not count `Exempt`. A check at 2026-11-03T05:18Z downloaded
`https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=NY&years=2024` and
compared it byte for byte with the frozen raw file; the two were identical. The
same check rebuilt the three partition CSVs into a temporary directory from that
download and the retained `split_v1.json`, and reproduced the hashes above. This is
a one-time observation: nothing states that the endpoint keeps a fixed version, so
a later download must still pass the hash check.

### 5.4 Preprocessing

The [data contract](docs/handbook/contracts/data_contract.md) and the
[classical](docs/handbook/contracts/preprocessing_classical.md) and
[neural](docs/handbook/contracts/preprocessing_neural.md) preprocessing
contracts own these rules. The order is:

1. Select the cohort and derive `y_approved` (`DatasetBViewBuilder` in
   `src/dataset_b/data/dataset_view.py`): 383,614 to 293,301 rows.
2. Assign partitions from the retained `artifacts/dataset_b/split_v1.json`:
   205,310, 43,995, and 43,996 rows. `write_split_csvs` in
   `src/dataset_b/workflows/create_split.py` writes the partition CSVs from a
   loaded manifest. Do not run that script's command-line entry point to obtain
   this split: it generates a new manifest from raw data, and `--force` would
   replace the locked file.
3. Apply the input-quality gate to the ordered 15 model inputs
   (`DatasetBInputQualityGate` in `src/dataset_b/data/split_inputs.py`). A blank
   rejects the row; it is never imputed or given a `MISSING` category. A direct
   count on 2026-11-03 found no contract-defined blank in these inputs in any of
   the three partition CSVs. The retained Validation and Test prediction files of
   all six final runs hold 43,995 and 43,996 rows.
4. Normalize raw `1111` to `Exempt` in the five exemption-coded flags only, inside
   that gate; `income = 1111` stays numeric.
5. Build the 19-column semantic frame (`numeric_two_channel.py` and
   `median_imputer.py` in `src/dataset_b/data/`): four numeric value channels, four
   status channels, and 11 categorical fields. Numeric placeholders use a median
   learned only from the permitted fit population.
6. Apply the model-specific tail below (`src/dataset_b/models/classical/profiles.py`
   and `neural_profiles.py`, `realmlp_native.py`, and `tabmmini_native.py` in
   `src/dataset_b/preprocessing/`). Validation and Test never contribute fitted state.

No single command runs these steps end to end. Fitted preprocessing state is retained
in the frozen run directories, for example `checkpoints/fitted_preprocessor.joblib`
for LR and `full_train/model/preprocessor.joblib` for TabMmini.

| Model                  | Numeric handling after step 5                                                                     | Model input                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Logistic Regression    | `log1p` on `income_value` and `loan_amount_value`, then `RobustScaler` on all four values | 62-column CSR                                                        |
| Random Forest, XGBoost | raw values with no log or scaling                                                                 | 62-column CSR                                                        |
| PlainMLP               | `log1p` on the same two values, then `RobustScaler` on all four values                        | 62-column CSR `float64`                                            |
| RealMLP                | `log1p` on the same two values                                                                  | 19-column native frame; numeric `float32`, 15 `category` columns |
| TabMmini               | `log1p` on the same two values; no project-level scaler or one-hot encoding                     | 19-column route; numeric `float32`, integer-indexed categories     |

### 5.5 Obtaining the assets

[`asset_manifest.json`](config/dataset_b/reproducibility/asset_manifest.json) declares
654 assets, 634 of them required, each with a size, SHA-256, role, and accepted run:

| Route                         | Assets | Handling by `download`                              |
| ----------------------------- | -----: | ----------------------------------------------------- |
| Historical HMDA source        |      1 | optional provenance; not fetched by the default route |
| Locked split CSVs             |      3 | downloaded directly, then verified by size and SHA-256 |
| Project bundles               |    647 | 11 ZIP files downloaded, verified, and extracted      |
| Git-tracked reference reports |      3 | already present in the clone                          |

`download` retries and resumes, verifies every hash, refuses to replace a local file
with a wrong hash, and rejects a share page returned instead of a file.

RQ3 `runtime.csv` remains in the accepted historical run directory, but it is excluded
from the core release inventory. It records non-deterministic wall-clock timings from
`time.perf_counter()`, is not an RQ3 scientific input, and is not compared by the
reproduction route. The manifest records this exclusion under `source_provenance`,
along with the source revision and a digest of the included source inventory.

**Status:** Phase C rebuilt and verified all 11 local ZIPs on 2026-11-09. The bundle
identities and 654-asset inventory were unchanged from the prebuild manifest; only
`source_provenance.source_revision` records direct-download implementation commit
`0aa028a00845852d383e0a180ccae82942bafad2`. Final evidence is in
`reports/reproducibility/phase_c/phase_c_bundle_verification_20261108T181358Z.json`.
Release commit `1d384482f7b00f84c06cdecbb4abd13380da91a2` and all 11 core LFS
objects were pushed to `codex/cuda-phase-b-c-handoff`; `git lfs fsck` passed before
and after the push. The default data route does not rebuild the splits from the live
HMDA endpoint. It downloads the three locked CSVs to `.part` files and publishes them
only after their declared size and SHA-256 pass; an existing correct file is skipped
and an existing incorrect file is never overwritten. The owner has not yet supplied
public bundle links or the three locked split-CSV URLs. Until then `download` exits 3,
and no anonymous download through those links has been verified. To record bundle
links, copy
[`bundle_links.example.json`](config/dataset_b/reproducibility/bundle_links.example.json)
to `bundle_links.json` in the same folder, fill in the 11 links, share each file with
anyone who has the link, and run:

```powershell
python -m src.dataset_b.reproduce maintainer-bind-links --links-file config/dataset_b/reproducibility/bundle_links.json
```

For the three split CSVs, copy
[`split_links.example.json`](config/dataset_b/reproducibility/split_links.example.json)
to `split_links.json`, replace all placeholders with the exact HTTPS links, and run:

```powershell
python -m src.dataset_b.reproduce maintainer-bind-split-links --links-file config/dataset_b/reproducibility/split_links.json
```

The locked identities are 23,631,341 bytes / `04575b2f...d692477` for Train,
5,063,935 bytes / `cd233414...9d3cbb` for Validation, and 5,064,953 bytes /
`5815caf7...2cae05` for Test. Raw HMDA remains historical provenance and is not a
prerequisite for frozen-checkpoint evaluation.

### 5.6 Optional RQ4 audit evidence

The local `scratch/` tree is a development and review workspace. It is ignored by
Git, is not an input to the core reproduction route, and is not a substitute for the
frozen RQ4a/RQ4b production artifacts under `artifacts/`. The production bundles and
the core [`asset_manifest.json`](config/dataset_b/reproducibility/asset_manifest.json)
therefore contain no `scratch/` destination.

A curated historical subset is described separately by
[`rq4_audit_evidence_manifest.json`](config/dataset_b/reproducibility/rq4_audit_evidence_manifest.json).
Its optional archive, `dataset_b_rq4_audit_evidence_20261105.zip`, preserves the
original repository-relative `scratch/...` paths, per-file sizes and SHA-256 values.
It excludes pytest basetemp/fixture trees, review scratch roots without saved review
logs, and all core production artifacts. This archive supports provenance review only;
it is not downloaded or verified by the core reproduction CLI and its absence does
not change any RQ result or core reproduction status.

The archive was assembled locally from the retained evidence without deleting,
moving, renaming, or overwriting any source file. The owner can publish that exact ZIP
separately and record its external URL using
[`rq4_audit_evidence_links.example.json`](config/dataset_b/reproducibility/rq4_audit_evidence_links.example.json).
The manifest currently leaves `public_url` null, so no public-distribution claim is
made.

## 6. Configuration and hyperparameters

Run `python -B -m src.dataset_b.cli validate-config` to validate the eleven
logical targets. The [Dataset B catalog](config/dataset_b/catalog.yaml) is a
router; scientific values remain in the target YAML files.

| Family   | Authoritative config                        | Locked highlights                                                                      | Search status                   |
| -------- | ------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------- |
| LR       | `config/dataset_b/model_protocol_v1.yaml` | L2, `C=1`, lbfgs, max_iter 1000, balanced class weight                               | fixed; no search                |
| RF       | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, gini, sqrt features, balanced class weight, seed 42                         | fixed; no search                |
| XGBoost  | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, depth 6, learning rate 0.3, hist, seed 42                                   | fixed; no search                |
| PlainMLP | `config/dataset_b/plain_mlp_v1.yaml`      | 256/128/64 ReLU, AdamW 0.001, dropout 0.1, deterministic algorithms, seed 42           | locked configuration; no search |
| RealMLP  | `config/dataset_b/realmlp_v1.yaml`        | three 256-unit layers, SELU, Adam, fixed epoch selected from Train-only pilot, seed 42 | locked configuration; no search |
| TabMmini | `config/dataset_b/tabmmini_v1.yaml`       | k=32, two 512 blocks, AdamW 0.002, fixed epoch from Train-only pilot, seed 42          | locked configuration; no search |

Values below come from the same files and from each run's stored results:

| Setting                    | PlainMLP                                                    | RealMLP                                | TabMmini                                  |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------- | ----------------------------------------- |
| Learning rate and schedule | 0.001                                                       | 0.04, `coslog4`                      | 0.002                                     |
| Weight decay               | 0.0001                                                      | 0.02, `flat_cos`                     | 0.0003                                    |
| Dropout                    | 0.1                                                         | 0.15, `flat_cos`                     | 0.1                                       |
| Batch size                 | 256, first size that passed the smoke run                   | 256                                    | 256, first size that passed the smoke run |
| Epochs                     | up to 100, patience 10, best epoch restored; final epoch 19 | up to 256 in the pilot; fixed epoch 50 | fixed epoch 55 from the pilot             |
| Mixed precision            | off; `float64`                                            | `NOT_RECORDED` here                  | `amp: false`                            |
| Gradient accumulation      | no such setting                                             | no such setting                        | no such setting                           |
| Pretrained weights         | none                                                        | none                                   | none                                      |

No hyperparameter search ran, so no search-results file exists. The fixed epochs come
from Train-only pilots: `pilot/decision.json` for RealMLP and `pilot/pilot_result.json`
for TabMmini, under each run root. Validation and Test were not used to choose them.

RQ2, RQ3, RQ4a, and RQ4b use their respective YAML files routed by the catalog.
The accepted results must be read with the exact config and run identity stored
in their manifests; current YAML status fields do not supersede STATUS when a
frozen config was intentionally left unchanged for lineage.

## 7. Training

No training was run for this guide, and the owner prohibited it on 2026-11-03. This
section first records only training that accepted historical evidence says was
completed; it then separates unrerun procedures so that a runbook is never presented
as runtime evidence. Accepted fits took far less than 4 GPU hours (Section 3.1), so
the guide template does not require a reduced configuration. Frozen outputs are under
each run root in Section 5.2, with `checkpoints/` or `full_train/` for models and
`reporting/` for tables.

### 7.1 Accepted completed-run inventory

| Scope              | Accepted completed runs                                                                   | Recorded seed scope                                                                  | Evidence-backed completion                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Classical RQ1      | `lr_final_15f_20261002_01`, `rf_final_15f_20261002_01`, `xgb_final_15f_20261003_01` | split/CV seed 42; RF/XGBoost random state 42                                         | five fresh Train-only CV fits and a fresh full-Train refit per family; frozen checkpoint and Test package retained |
| PlainMLP RQ1       | `plain_mlp_final_15f_20261004_03`                                                       | seed 42; deterministic-algorithm policy recorded                                     | Train-only selection/CV and fresh full-Train fit retained as accepted historical evidence                          |
| RealMLP RQ1        | `realmlp_final_15f_20261004_01`                                                         | seed 42; fixed epoch 50 selected from the Train-only pilot                           | five fresh Train-only CV fits and a fresh full-Train fit retained as accepted historical evidence                  |
| TabMmini RQ1       | `tabmmini_final_15f_20261004_01`                                                        | pilot/CV seed 42; fixed accepted full-Train epoch 55                                 | Train-only pilot/CV and fresh full-Train fit retained as accepted historical evidence                              |
| RQ4b-3 development | `rq4b3_adv_20261019_02`, `rq4b3_adv_20261021_01`, `rq4b3_adv_20261022_01`           | lambdas 0 and 0.01 at seeds 42/43/44; screening lambdas 0.03/0.1/0.3 at seed 42 only | nine development units are recorded complete; Gate C selected lambda 0; closed evidence is not rerun               |

This inventory does not require or authorize fresh training. It reports only
completed accepted runs and their actual seed coverage; it does not upgrade a
single-seed screening arm into multi-seed evidence.

### 7.2 Documented procedure routes

The following routes are procedures, not current runtime PASS claims.

| Family           | Procedure                                                                                                                                                                                                                                          | Classification                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| LR               | [baseline training runbook](docs/handbook/runbooks/baseline_training.md); fold inputs with `python -m src.dataset_b.workflows.export_classical_cv --run-id $RUN_ID`; fit through `notebooks/dataset_b/final/models/01_linear_model_final.ipynb` | `DOCUMENTED_NOT_RERUN`; there is no repository-level LR training command                                           |
| RF               | same fold inputs, then `python -m src.dataset_b.models.classical.rf.training run --run-id $RUN_ID`                                                                                                                                               | `DOCUMENTED_NOT_RERUN`                                                                                             |
| XGBoost          | same fold inputs, then `python -m src.dataset_b.models.classical.xgboost.training run --run-id $RUN_ID`                                                                                                                                          | `DOCUMENTED_NOT_RERUN`                                                                                             |
| PlainMLP         | [neural training runbook](docs/handbook/runbooks/neural_training.md) plus `plain_mlp_v1.yaml`; `python -m src.dataset_b.models.neural.plain_mlp.training cv run --run-id $RUN_ID`, then `full-train run` with the same option                 | `ACCEPTED_HISTORICAL_EVIDENCE` for the frozen run; a new run is `DOCUMENTED_NOT_RERUN` and separately authorized |
| RealMLP          | runbook procedures for pilot, CV, and full-Train refit; `python -m src.dataset_b.models.neural.realmlp.cv run --run-id $RUN_ID` and `python -m src.dataset_b.models.neural.realmlp.full_train run --run-id $RUN_ID`                            | `NOT_PUBLIC_HANDOVER_COMMAND`; operator/internal route, not executed here                                          |
| TabMmini         | [TabMmini training runbook](docs/handbook/runbooks/tabmmini_training.md) and locked config; `run_tabmmini_full_train` in `src/dataset_b/models/neural/tabmmini/training.py`                                                                     | `ACCEPTED_HISTORICAL_EVIDENCE` for the frozen run; no command-line training route                                  |
| RQ4b adversarial | staged internal route in the neural runbook, one authorized unit at a time                                                                                                                                                                         | `NOT_PUBLIC_HANDOVER_COMMAND`; closed evidence must not be rerun                                                   |

All families use five fresh Train-only CV fits and a fresh full-Train refit under
their owning contract. Validation selects the threshold; Test is final. A new
execution requires a new run ID and may not overwrite a frozen run.

The neural runbook still names `src.dataset_b.runtime.realmlp_cv` and
`realmlp_full_train`. Those modules are compatibility shims with no command-line entry,
so running them does nothing; use the commands above.

Neural checkpoint and recovery behavior: PlainMLP restores its best inner-stopping epoch;
RealMLP and TabMmini train to a fixed epoch. There is no `--resume` option, and TabMmini
records resuming as disallowed. Work happens in a run-scoped `working/` directory and
publishes as a whole (Section 10 of the neural runbook), so an interrupted run does not
reach `final/`.
Epoch history is stored per family: RealMLP in `training_history.json` and `.jsonl` for its
pilot, folds, and full fit; PlainMLP in those files for its pilot and as a result field
for its fits; TabMmini as a `history` key in `pilot/pilot_result.json`. No external
tracking tool is used. Peak training VRAM is `NOT_RECORDED` for the final fits.

## 8. Evaluation and inference

Evaluation and inference run through `src.dataset_b.reproduce`. The navigation CLI
`src.dataset_b.cli` deliberately stays read-only.

```powershell
python -m src.dataset_b.reproduce evaluate --model all
python -m src.dataset_b.reproduce compare
python -m src.dataset_b.reproduce research --route rq3
```

`evaluate` accepts `logistic_regression`, `random_forest`, `xgboost`, `plain_mlp`,
`realmlp`, `tabmmini`, or `all`. For each model it verifies asset hashes, the accepted
run identity, the frozen Validation threshold, and the ordered Test records; it then
runs the frozen checkpoint once and computes ten metrics and the 1,000 locked bootstrap
replicates. Neural models stop rather than fall back to CPU. Each model writes
`test_predictions.csv`, `rq1_test_metrics.json`, `rq1_bootstrap_metrics.csv`, and
`receipt.json` under `reports/reproducibility/rq1/<model>/`. Confusion counts are in the
metrics file. `compare` writes `reports/reproducibility/rq1/comparison.csv`.

Observed on 2026-11-03 on the author machine: all six models evaluated, and 402 of 402
comparison rows passed, with a largest prediction difference of 2.2e-16 for Logistic
Regression and 0 for the other five. No fit ran; this rests on source inspection,
because the receipt counters are fixed values rather than runtime measurements.

| RQ route | Classification                                                                                    | Status on 2026-11-03 |
| -------- | ------------------------------------------------------------------------------------------------- | -------------------- |
| `rq2`  | checkpoint-based; CPU frozen inference over Validation, no Test                                   | not run              |
| `rq3`  | artifact-based; zero fit and zero prediction calls                                                | 5 of 5 rows passed   |
| `rq4a` | report-only; validates the stored packages without fitting the post-processor                     | 4 of 4 rows passed   |
| `rq4b` | artifact-based; Test and applicant-sex tables recomputed from saved predictions; Gate C read only | 17 of 17 rows passed |

To score new records with a frozen checkpoint, supply a CSV with `record_id` and the 15
raw model inputs. Nothing is fitted; the input-quality gate rejects blank or unknown
values, and the frozen threshold sets the decision:

```powershell
python -m src.dataset_b.reproduce predict --model logistic_regression --input new_records.csv --output scores.csv
```

Example input:

```text
record_id,loan_type,loan_purpose,lien_status,reverse_mortgage,open-end_line_of_credit,business_or_commercial_purpose,income,debt_to_income_ratio,loan_amount,loan_to_value_ratio,loan_term,occupancy_type,preapproval,interest_only_payment,balloon_payment
dataset_b-000264298,1,2,2,2,1,2,88,20%-<30%,75000.0,42.2,360,1,2,2,2
```

Expected output, taken from the accepted Logistic Regression Test predictions for this
record:

```text
record_id,p_approved,threshold,predicted_approved
dataset_b-000264298,0.5359965198181446,0.493,1
```

`predict` is covered by synthetic unit tests and was not run on a frozen checkpoint.
There is no demo application. Stored predictions in a run root are references for
comparison, not substitutes for this route.

### 8.1 RQ1 metric definitions

The [split and evaluation contract](docs/handbook/contracts/split_and_evaluation.md)
owns the required outputs. Scores are `p_approved = P(y_approved = 1)`, and a row
is predicted Approved when `p_approved >= threshold`. In the table, Approved is the
positive class: `TP` and `FN` are true Approved rows predicted Approved and Denied;
`TN` and `FP` are true Denied rows predicted Denied and Approved. Each model's threshold
maximizes Balanced Accuracy over the 999 Validation candidates `0.001` to `0.999`;
ties go to the smallest distance from `0.5`, then the lower threshold. The formulas
below are read from the LR route, `src/dataset_b/models/classical/lr/test_evaluation.py`.
Each other family has its own evaluation module, which this table does not restate.

| Metric key                   | Definition                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `roc_auc_approved`         | `sklearn.metrics.roc_auc_score(y_approved, p_approved)`                                                    |
| `average_precision_denied` | `sklearn.metrics.average_precision_score(1 - y_approved, 1 - p_approved)`                                  |
| `precision_denied`         | `TN / (TN + FN)`                                                                                           |
| `recall_denied`            | `TN / (TN + FP)`                                                                                           |
| `f1_denied`                | `2 * TN / (2 * TN + FN + FP)`                                                                              |
| `recall_approved`          | `TP / (TP + FN)`                                                                                           |
| `balanced_accuracy`        | mean of `recall_approved` and `recall_denied`                                                            |
| `brier_score_approved`     | mean of `(p_approved - y_approved)^2`; stored as `brier_approved` in the LR Test evaluation module       |
| `calibration_intercept`    | offset `a` that solves `mean(expit(logit(p) + a) - y) = 0`, with `p` clipped to `[1e-15, 1 - 1e-15]` |
| `calibration_slope`        | slope `b` from the logistic fit of `y` on `a + b * logit(p)`                                           |

## 9. Seeds and determinism

| Scope         | Locked policy                                                                 | Evidence/limitation                                                 |
| ------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Split         | seed 42; 70/15/15 stratified by `y_approved`                                | immutable `split_v1`                                              |
| RQ1 CV        | five-fold StratifiedKFold, shuffle, seed 42, Train only                       | accepted six-model run lineage                                      |
| LR/RF/XGBoost | fixed configuration; RF/XGBoost random_state 42                               | no hyperparameter search                                            |
| PlainMLP      | seed 42; deterministic algorithms true                                        | GPU identity/version changes may still alter numerical behavior     |
| RealMLP       | seed 42; five fresh folds; fixed epoch 50 from Train-only pilot               | deterministic cross-host identity is not claimed                    |
| TabMmini      | seed 42; pilot and CV seed 42; fresh state                                    | fixed epoch 55 in accepted full-Train result                        |
| RQ2           | sample/background/assignment seeds 4201/4202/4203; bootstrap seed 4204        | 1,000 shared replicates                                             |
| RQ4b          | model seeds 42, 43, 44 for lambdas 0 and 0.01; screening lambdas only seed 42 | three-seed summaries are not pooled with within-seed bootstrap rows |

**Acceptance threshold.** A reproduced RQ1 result matches when it meets the policy in
[`numerical_policy.json`](config/dataset_b/reproducibility/numerical_policy.json), which
is stricter than the course guideline of 0.5 points for traditional ML and 1 point for
deep learning. Replicate IDs and validity payloads must match exactly;
`calibration_slope` is compared after rounding to six decimals; other bootstrap metrics
and CI endpoints use absolute tolerance `1e-14`; point metrics use `1e-12`. These values
come from Section 7 of the split and evaluation contract, and the file holds one entry
per model so that any family can differ. The LR, RF, XGBoost, and PlainMLP evaluation
modules use the same bootstrap and slope values; the RealMLP and TabMmini review modules
were not compared. The `1e-12` limit on predictions and on RQ2-RQ4 tables is not locked
by a contract; it was fixed before those comparisons. Exact SHA-256 is for
stored-artifact integrity and lineage, not a universal floating-point equivalence rule.

**Seed setting.** Seed handling differs by family, and there is no shared `set_seed`
function. No fit runs in the reproduction route.

| Family      | Seeded sources                                                                                                                                                                                                                                               | Deterministic settings                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| LR          | none; `lbfgs` takes no model seed                                                                                                                                                                                                                          | not applicable                                                           |
| RF, XGBoost | `random_state = 42`                                                                                                                                                                                                                                        | `tree_method: hist` for XGBoost                                        |
| PlainMLP    | Python `random`, NumPy, torch, and CUDA, from `random_state = 42`                                                                                                                                                                                        | deterministic algorithms and cuDNN deterministic on; cuDNN benchmark off |
| RealMLP     | `random_state = 42` passed to PyTabKit                                                                                                                                                                                                                     | not set in the project code read                                         |
| TabMmini    | config `seed: 42`; the fit route's data order uses a generator seeded `42 + epoch`; `torch.manual_seed(42)` appears only in the smoke attempt, and no explicit global torch seed for model initialization was found in the `src/dataset_b` fit route | not set in the project code read                                         |

Project code sets no `worker_init_fn`; PyTabKit internals were not inspected. Results may differ on another GPU or CUDA version; the
2026-11-03 comparison used the GPU model recorded by the accepted runs.

Each of the six final RQ1 models has one final fit. RF, XGBoost, PlainMLP, and RealMLP
use model seed 42; TabMmini carries config `seed: 42` as the table above describes; LR's
`lbfgs` configuration has no model seed. All six use the same split and fold seed 42. Five-fold CV reports variation across folds, not
across seeds, so no multi-seed mean and standard deviation exists for RQ1. Producing
one would require new training, which has not been authorized. RQ4b lambdas 0 and
0.01 are the only arms with three seeds.

## 10. Result to report mapping

This mapping is inspectable without notebook execution.

| Reporting scope                          | Notebook/report                                                                                                                    | Accepted run/config                                            | Procedure                                                              | Durable output                                              | Classification                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| RQ1 six-model metrics                    | `notebooks/dataset_b/final/models/01_...06_*.ipynb`; `07_model_total_review.ipynb`; `reports/RQ1_Research_Metrics_Final.txt` | six frozen RQ1 run IDs; model configs in Section 6             | final notebook consumes frozen evidence                                | per-model reporting directories and RQ1 report              | `ACCEPTED_HISTORICAL_EVIDENCE`                                  |
| RQ2 explanations                         | `08_rq2_gsv_sl_results.ipynb`; `reports/RQ2_Research_Metrics_Final.txt`                                                        | `rq2_cpu_production_20261009_01`; `rq2_protocol.yaml`      | CPU frozen inference; 256 explanation rows; 1,000 bootstrap replicates | `artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01/` | `ACCEPTED_HISTORICAL_EVIDENCE`                                  |
| RQ3 fairness                             | `09_rq3_fairness_results.ipynb`; combined RQ3/RQ4 report                                                                         | `rq3_fairness_20261010_01`; `rq3_protocol.yaml`            | artifact-only audit; zero fit/prediction calls                         | `artifacts/dataset_b/rq3/rq3_fairness_20261010_01/`       | `ACCEPTED_HISTORICAL_EVIDENCE`                                  |
| RQ4a post-processing                     | `11_rq4a_equalized_odds_results.ipynb`; combined RQ3/RQ4 report                                                                  | `rq4a3_postproc_20261017_01`, `rq4a6_postproc_20261017_01` | saved-prediction post-processing                                       | run reporting directories                                   | `ACCEPTED_HISTORICAL_EVIDENCE` with documented G6 limitations   |
| RQ4b adversarial                         | `12_rq4b_adversarial_results.ipynb`; Figure 3 report output                                                                      | three development runs, Gate C, and `rq4b3_test_20261024_01` | staged internal procedure; no rerun                                    | Test/package reporting artifacts                            | `ACCEPTED_HISTORICAL_EVIDENCE`; interpretation remains separate |
| RQ4b sex supplement                      | `13_rq4b_applicant_sex_supplementary_results.ipynb`                                                                              | `rq4b3_sex_supplementary_20261024_02`                        | evaluation from saved predictions                                      | supplementary package                                       | `ACCEPTED_HISTORICAL_EVIDENCE`                                  |
| Publication figures and assembled report | `10_publication_figures.ipynb`, `14_research_report.ipynb`, `reports/paper_figures/`                                         | inputs above                                                   | read/report procedure                                                  | PDFs and verification notes listed by `report-index`      | `DOCUMENTED_NOT_RERUN` in this task                             |

The retained reports provide object-level provenance beneath this scope map:

- the RQ1 report records the six-model metric and bootstrap sources;
- the RQ2 report records its contract, config, accepted run, and notebook;
- the combined RQ3/RQ4 report includes Tables 2A, 2B, 5A, A1, and A2 with
  accepted run and source-artifact lineage;
- `reports/paper_figures/verification_notes.md` records the interpretation
  boundaries for Figures 2, 3, 5, and 6.

Notebook paths above are under `notebooks/dataset_b/final/`. The migration to
that directory was path-only; it did not execute or regenerate notebook output.

### 10.1 Object-level mapping

`python -m src.dataset_b.reproduce reports` writes this mapping as
`reports/reproducibility/reports/report_mapping.csv` for the objects it covers.
Sources for the text-report tables come from each table's title and from Table A2; they
were not checked cell by cell. The text reports were assembled by authored report tasks,
so they have no regeneration command. Notebook commands below were not executed and need
`jupyter`, which no profile lock includes; install it separately. The combined report
uses the number 5A for two tables, matched deltas and per-unit metrics; the row below
covers both.

| Object                                                                                  | Reproduction command                                                                                                                                      | Config                                         | Source or generated output                                                                            | Main value or status                           |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| RQ1 ten-metric table, Sections 5 and 11 of the RQ1 report                               | `reproduce evaluate --model all`, then `reproduce reports`                                                                                            | model configs in Section 6                     | `reports/reproducibility/reports/rq1_metrics_long.csv`                                              | Section 1.0 values; matches accepted reporting |
| RQ1 report Section 6: CV folds, confusion matrix, calibration curve                     | none in the reproduction CLI; confusion counts are in `rq1_test_metrics.json`                                                                           | model configs in Section 6                     | each run's `reporting/cv_fold_metrics.csv`, `confusion_matrix.csv`, and `calibration_curve.csv` | not regenerated                                |
| Other RQ1 and RQ2 report sections: scope, lineage, cohort, checks, ledgers, limitations | none; authored report text                                                                                                                                | Sections 5 and 6                               | `reports/RQ1_Research_Metrics_Final.txt`, `reports/RQ2_Research_Metrics_Final.txt`                | not regenerated                                |
| RQ1 paired comparisons, Section 7 of the RQ1 report                                     | `07_model_total_review.ipynb`                                                                                                                           | `split_and_evaluation.md`                    | six stored bootstrap tables                                                                           | 150 rows; not regenerated                      |
| RQ2 report Sections 5 to 8                                                              | `reproduce research --route rq2`                                                                                                                        | `rq2_protocol.yaml`                          | `reports/reproducibility/research/rq2/`                                                             | 28 of 28 comparison rows passed                |
| RQ3 group, gap, and bootstrap tables                                                    | `reproduce research --route rq3`                                                                                                                        | `rq3_protocol.yaml`                          | `reports/reproducibility/research/rq3/output/`                                                      | 5 of 5 comparison rows passed                  |
| Tables 4A to 4E                                                                         | `reproduce research --route rq4a`                                                                                                                       | `rq4a_postprocessing.yaml`                   | stored `reporting/` of both RQ4a runs                                                               | stored package validated; not regenerated      |
| Tables 5A to 5E                                                                         | `reproduce research --route rq4b`                                                                                                                       | `rq4b_adversarial.yaml`                      | `reports/reproducibility/research/rq4b/output/`                                                     | 17 of 17 comparison rows passed                |
| Tables 5F, 5G, 5H                                                                       | none; supplementary tables and closed development evidence                                                                                                | `rq4b_adversarial.yaml`                      | RQ4b Test and development runs                                                                        | not regenerated                                |
| Gate C selection                                                                        | none; closed evidence                                                                                                                                     | `rq4b_adversarial.yaml`                      | `gate_c_selection.json`                                                                             | `selected_lambda = 0.0`                      |
| Tables 2A, 2B, A1, A2 of the combined report                                            | none; assembled report tables                                                                                                                             | Sections 6 and 10                              | `reports/RQ3_RQ4a_RQ4b_Research_Metrics_Final.txt`                                                  | not regenerated                                |
| Figure 1                                                                                | none; `figure_index.json` marks it `UNCHANGED_OUTSIDE_NOTEBOOK`                                                                                       | none                                           | not in `reports/paper_figures/`                                                                     | no regeneration route                          |
| Figures 2 and 3                                                                         | `jupyter nbconvert --to notebook --execute notebooks/dataset_b/final/reporting/14_research_report.ipynb --output-dir reports/reproducibility/notebooks` | `reports/paper_figures/source_manifest.json` | `reports/paper_figures/`                                                                            | present; not regenerated                       |
| Figures 4, 5, and 6                                                                     | same command                                                                                                                                              | `reports/paper_figures/source_manifest.json` | indexed in `figure_index.json`; **files absent from this repository**                         | not regenerated                                |
| Publication figures under `reports/figures/publication/`                              | same command with `10_publication_figures.ipynb`                                                                                                        | notebook inputs                                | `reports/figures/publication/`                                                                      | not regenerated                                |

`reports` exits 1 with `INCOMPLETE` while any object it covers is `NOT_RUN` or
`NOT_REGENERATED`.

## 11. Limitations and troubleshooting

| Symptom or limitation                                                                                               | Meaning                                                                                                                         | Required action                                                                                      |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Clean clone lacks raw data/artifacts/checkpoints                                                                    | retained evidence is not Git-distributed                                                                                        | run `download` after the links in Section 5.5 are recorded                                         |
| `download` exits 3 with `required assets have no public download route`                                         | the bundle links are not recorded yet                                                                                           | follow Section 5.5                                                                                   |
| `Link returned an HTML page instead of a file`                                                                    | the Drive file is not shared publicly, or Drive returned a notice page                                                          | share the file with anyone who has the link, then rerun `download`                                 |
| `git clone` asks for credentials or returns 401                                                                   | anonymous repository access returned HTTP 401 on 2026-11-03                                                                     | request repository access from the group                                                             |
| `ModuleNotFoundError` when starting the CLI                                                                       | the host environment from Section 1.0 is not active                                                                             | activate `.venv` and install its lock                                                              |
| `uv venv` fails once with `Missing expected target directory`                                                   | observed right after `uv` first downloaded a Python build                                                                     | `environment --install` retries once; run it again if needed                                       |
| CUDA out of memory                                                                                                  | the accepted batch size may not fit a smaller GPU                                                                               | close other GPU work; do not change locked settings or fall back to CPU                              |
| Clean dependency installation fails                                                                                 | a clean-machine run was not executed                                                                                            | record OS, Python, resolver output, package index, GPU/driver/CUDA; do not edit frozen evidence      |
| CUDA unavailable                                                                                                    | CUDA-only route is not runnable on that host                                                                                    | stop; do not silently fall back to CPU                                                               |
| Raw CSV hash differs                                                                                                | it is not the exact historical project export                                                                                   | preserve the mismatch and do not relabel it as the frozen source                                     |
| Frozen hash fails                                                                                                   | stored lineage differs from the accepted lock                                                                                   | stop and investigate; do not regenerate or overwrite                                                 |
| `validate-artifacts` reports a locked hash mismatch for `split_v1.json` or `rq1_test_bootstrap_manifest.json` | the local copy differs from the LF bytes pinned by the accepted run manifests; a CRLF-converted copy has 8,212,697 or 917 bytes | obtain the file through `download`; do not convert or overwrite accepted files                     |
| Multi-seed RQ1 mean and standard deviation are absent                                                               | each of the six final RQ1 models has one final fit                                                                              | read Section 9; do not present fold variation as seed variation                                      |
| Notebook contains saved error output                                                                                | it is not clean saved reporting evidence                                                                                        | inspect the named notebook as JSON; do not execute or resave without separate authorization          |
| Historical `D:\Proj\...` or `/content/drive/MyDrive/...` path appears                                           | it is run-profile lineage, not a portable repository default                                                                    | interpret against the recorded runtime; use repository-relative catalog paths for current navigation |

The frozen configuration and manifests intentionally preserve historical
absolute paths. Do not edit them for portability. Current navigation resolves
the project root from source and requires catalog targets to be
repository-relative. The Colab profile's `/content` and Google Drive roots are
profile-specific and its recorded runtime execution remains not started.

### 11.1 Current-source runtime verification gaps

- The clean-clone sequence and download through the Drive links were not executed.
- The qualified-host route was run in place, not from a clean clone.
- Public download through the missing direct split-CSV URLs was not verified.
- Three notebook/authored-report objects remain `NOT_REGENERATED`; `reports` therefore
  returns the documented `INCOMPLETE` status.

These are conditions still needed for full current-source runtime reproducibility.

### 11.2 Intentionally not rerun under the accepted scope

On the owner's decision of 2026-11-03, no model was trained or fitted and no further
process was rerun. The RQ4a route validates stored packages without regenerating its
post-processor. RQ1 retains one final fit per model and therefore has no multi-seed
result (Section 9).

The ignored `scratch/` workspace remains local and was not deleted. A curated optional
RQ4 audit bundle was created separately from the core asset manifest; publication of
that bundle remains an owner delivery action and is not a runtime-reproduction gate.

### 11.3 Report and artifact regeneration gaps

Figures, text reports, and notebooks were not regenerated. Figures 4 to 6 are absent
from this repository, Figure 1 has no regeneration route, and some authored report
tables have no regeneration command. These are limitations for full report
regeneration; they do not change the accepted frozen experimental evidence.

The version 2.3 documentation review did not itself perform a new clean installation,
training run, notebook execution, or end-to-end clean-clone run. Historical
author-machine observations are recorded separately in Section 1 and
[STATUS](docs/handbook/STATUS.md). Cross-host determinism, driver/cuDNN compatibility,
RAM sufficiency, and end-to-end train-from-scratch remain unverified.

### 11.4 Current-host technical resolution

On 2026-11-05 the owner closed the current Phase B and Phase C attempt. Resolution of
the issue encountered in that attempt is `COMPLETE`: the local asset gate passed
654/654, and the evaluation command stopped because this machine has no CUDA and does
not contain the five locked virtual environments.

The owner records that the locked virtual environments are on another machine. The
stopped current-host attempt is an environment-boundary record, not a checkpoint,
metric, or scientific-result failure. `compare`, RQ2, `reports`, clean-clone execution,
public download verification, final tagging, and release publication were not run.
Completion here applies only to diagnosis, recording, and closure of this technical
issue; it does not state that the entire reproduction runtime is complete.

## 12. License citation and AI declaration

The owner completes the blank license, data terms, and citation fields below.

| Item                                    | Repository-backed status                                                                                                                                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project code license                    |                                                                                                                                                                                                                                                                                                               |
| HMDA data license/terms                 |                                                                                                                                                                                                                                                                                                               |
| Project citation                        |                                                                                                                                                                                                                                                                                                               |
| Model/library citations                 |                                                                                                                                                                                                                                                                                                               |
| Package versions                        | recorded in `environments/reproducibility/` and in configs/manifests                                                                                                                                                                                                                                        |
| Libraries used                          | [scikit-learn](https://github.com/scikit-learn/scikit-learn), [XGBoost](https://github.com/dmlc/xgboost), [PyTorch](https://github.com/pytorch/pytorch), [PyTabKit](https://github.com/dholzmueller/pytabkit), [TabM](https://github.com/yandex-research/tabm), [Fairlearn](https://github.com/fairlearn/fairlearn) |
| Generative AI evidence                  | repository history and handbook record historical Claude implementation work and Codex implementation/review/documentation work; authorship and review roles must not be relabeled                                                                                                                            |
| Generative AI on the reproduction route | Claude Code implemented `src/dataset_b/reproduce/`, the environment profiles, and this guide's 2026-11-03 revisions; checks were unit tests, hash and numerical comparison, and a read-only verifier agent                                                                                                  |

The project predicts approval behavior for the documented cohort. It does not
predict default risk or creditworthiness, and its outputs are not causal, legal,
or policy evidence of discrimination.
