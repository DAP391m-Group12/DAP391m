# Dataset B Documentation-Level Reproduction Guide

This guide explains how a new reader can locate the Dataset B inputs,
configuration, frozen evidence, reports, and model-specific procedures. Its
acceptance scope is documentation-level reproducibility: the documented paths,
commands, checksums, and ownership routes must be inspectable from the
repository. It does not guarantee that every computer can install or execute
all model backends because operating systems, Python builds, CPU/GPU hardware,
CUDA drivers, memory, and package availability differ.

Completed experiments and frozen evidence must not be rerun or overwritten
merely to verify the refactor. When the project publishes a release tag or a
submission commit, readers should record that Git identity alongside their
verification output instead of relying on an unversioned working tree.

## 1. Quick Start for Repository Verification

Run commands from the repository root. The installation step prepares the
general CPU-oriented environment described by `requirements.txt`; it is not a
replacement for a model-specific CUDA environment.

```text
git clone https://github.com/DAP391m-Group12/DAP391m.git
cd DAP391m
git rev-parse HEAD
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.dataset_b.cli status
.\.venv\Scripts\python -m src.dataset_b.cli validate-config
.\.venv\Scripts\python -m src.dataset_b.cli validate-artifacts --scope frozen
.\.venv\Scripts\python -m src.dataset_b.cli validate-notebooks --scope final
.\.venv\Scripts\python -m src.dataset_b.cli report-index
```

On Linux or macOS, replace `.\.venv\Scripts\python` with
`.venv/bin/python`. A successful documentation-level check reports:

- repository status `PASS` with Dataset B active;
- 11 valid Dataset B configuration targets;
- six frozen-model markers and three verified root locks;
- 14 final notebooks with no saved error payload;
- six indexed report outputs;
- zero checkpoint loads and zero inference calls.

Installation failure on a different host is an environment compatibility
issue, not evidence that the stored experiment results changed. Record the
Python version, platform, package resolver error, and GPU/CUDA details before
changing any pinned dependency.

## 2. Repository Routes

The main paths are:

```text
config/                  project and Dataset B configuration contracts
data/raw/                immutable HMDA source location
data/dataset_b/          Dataset B derived data and split material
artifacts/dataset_b/     frozen and working experiment evidence
notebooks/dataset_b/     historical and final reporting notebooks
src/dataset_b/           Dataset B implementation
docs/handbook/           method contracts, runbooks, and execution status
reports/                 indexed research metrics and figures
tests/                   unit, contract, and integration tests
```

Use these routers instead of guessing a model-specific path:

- [`README.md`](README.md) for project boundaries and the top-level tree;
- [`docs/handbook/README.md`](docs/handbook/README.md) for method contracts and
  runbooks;
- [`docs/handbook/STATUS.md`](docs/handbook/STATUS.md) for accepted execution
  state and run identities;
- [`config/README.md`](config/README.md) for configuration discovery;
- [`notebooks/dataset_b/README.md`](notebooks/dataset_b/README.md) for notebook
  roles;
- [`src/dataset_b/README.md`](src/dataset_b/README.md) for canonical source
  ownership and compatibility paths.

## 3. Environment Declaration

The repository declares Python `>=3.10`. The general requirements file pins
the packages needed by the current source, CPU-safe checks, tests, and notebook
support. Accepted historical runs record their actual environment in their
runtime manifests; those manifests remain the authority for reproducing a
specific run.

The general requirements route is CPU-oriented. Neural production routes may
require a CUDA-enabled PyTorch build and a compatible NVIDIA driver. The
repository deliberately does not prescribe one GPU model, CPU model, or fixed
amount of RAM/VRAM for every host. Before a GPU run, follow the applicable
model runbook and verify:

```text
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA unavailability must stop a CUDA-only production route; it must not cause
silent CPU fallback. Hardware-dependent runtime and numerical variation must
be reported with the host environment rather than hidden by changing frozen
results.

## 4. Data Source and Integrity

Dataset B uses the 2024 HMDA loan-level data for New York. The project source
was obtained through the official HMDA Data Browser and is expected at:

```text
data/raw/state_NY_2024.csv
```

Official source routes:

- [HMDA Data Browser 2024](https://ffiec.cfpb.gov/data-browser/maps/2024)
- [2024 national loan-level dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024)

The exact historical project file has:

```text
size_bytes: 145740132
sha256: e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d
```

Verify it without modifying the file:

```text
# Windows PowerShell
Get-FileHash -Algorithm SHA256 data/raw/state_NY_2024.csv

# Linux or macOS
sha256sum data/raw/state_NY_2024.csv
```

The raw CSV is intentionally not distributed as part of the handover commit.
If the exact historical export is unavailable, a newly downloaded HMDA file
may have a different hash or row order and must not be represented as the
frozen source. In that case, the repository still supports inspection of the
accepted frozen evidence, but regeneration of the historical split and model
results is outside this documentation-level guarantee.

The authoritative cohort, target, role, exclusion, and input-quality rules are
in [`docs/handbook/contracts/data_contract.md`](docs/handbook/contracts/data_contract.md).
The immutable split policy is in
[`docs/handbook/contracts/split_and_evaluation.md`](docs/handbook/contracts/split_and_evaluation.md)
and [`config/dataset_b/split_v1.yaml`](config/dataset_b/split_v1.yaml).

## 5. Frozen Split and Bootstrap Locks

The read-only artifact validator checks the following root locks:

| Path | SHA-256 |
|---|---|
| `artifacts/dataset_b/split_v1.json` | `65d15274095076dbd2306ac7c7b25cf02f1032ed4b2410b30e1d00e3f5b446ee` |
| `artifacts/dataset_b/rq1_test_bootstrap_indices.npz` | `2ee377b6fbae8a3a6453a7d8f2f4a0a3f1ce61374f1258ef882589b84230fee1` |
| `artifacts/dataset_b/rq1_test_bootstrap_manifest.json` | `7b202b9523820c4ec6b2e777adf1a672a43c6bd39ae42842fd52726d1e88a1cb` |

The split contract uses seed 42, a 70/15/15 Train/Validation/Test split,
stratification by `y_approved`, and five Train-only stratified CV folds. Test
must not be used for feature selection, preprocessing fit, hyperparameter
selection, or threshold selection.

## 6. Configuration and Model Procedures

Run `python -m src.dataset_b.cli validate-config` to resolve the 11 logical
configuration roles. Scientific values belong to the target YAML files, not
to the catalog or this guide.

Model-family procedures are intentionally separate:

- classical preprocessing and baselines: handbook classical contract and
  baseline training runbook;
- PlainMLP and RealMLP: neural preprocessing contract and neural training
  runbook;
- TabMmini: TabMmini v1 training runbook;
- RQ2, RQ3, and RQ4: their handbook contracts and accepted run identities in
  `docs/handbook/STATUS.md`.

There is no repository-wide public `train`, `evaluate`, or `predict` command.
Do not invent one. The five public CLI commands are read-only navigation and
integrity checks. Any model execution requires the applicable runbook,
environment, configuration, frozen lineage, and separate authorization.

## 7. Results and Report Mapping

Use the report index as the stable entry point:

```text
python -m src.dataset_b.cli report-index
```

It routes to the current final notebooks and these report outputs:

```text
reports/RQ1_Research_Metrics_Final.txt
reports/RQ2_Research_Metrics_Final.txt
reports/RQ3_RQ4a_RQ4b_Research_Metrics_Final.txt
reports/paper_figures/fig02_postprocessing_group_error_rates.pdf
reports/paper_figures/fig03_adversarial_group_error_rates.pdf
reports/paper_figures/verification_notes.md
```

The final notebooks consume frozen evidence and present the accepted results.
The handover refactor moved notebook paths without changing notebook bytes. It
did not execute notebooks or regenerate figures. Current execution status and
accepted run IDs must be read from `docs/handbook/STATUS.md`.

## 8. Reproduction Boundaries

Documentation-level reproduction is satisfied when a reader can identify the
source, verify available checksums, resolve configuration and method ownership,
inspect accepted results, and understand the correct execution route. It does
not assert that:

- every operating system or processor can install the same binary packages;
- every host has compatible CUDA hardware, drivers, or memory;
- GPU training is byte-identical across devices;
- the exact historical raw export can still be downloaded;
- notebooks can be rerun safely without the model-specific environment;
- completed or frozen experiments should be reopened.

When execution is required, record the host environment and follow the
model-specific tolerance, seed, threshold, split, and artifact rules. Never
overwrite accepted artifacts to make a new environment appear identical.

## 9. Known Troubleshooting Routes

| Symptom | Meaning | Action |
|---|---|---|
| CUDA is unavailable | The host cannot run a CUDA-only production route | Use the read-only checks or prepare a compatible GPU environment; do not silently fall back to CPU |
| `fairlearn` import fails | The RQ4a dependency is missing | Install the pinned requirements and recheck the import before any authorized RQ4a work |
| Lightning import fails | The neural callback/training dependency is missing | Install the pinned requirements; the source supports the recorded `pytorch_lightning` route |
| Raw CSV hash differs | The file is not the exact historical project export | Do not replace the frozen-source claim; document the new source separately |
| A frozen lock fails | Stored lineage differs from the accepted root lock | Stop and investigate; do not regenerate or overwrite the lock |
| A notebook has a saved error | The notebook is not clean reporting evidence | Inspect the named notebook without executing or resaving it |

## 10. Safety Statement

The project predicts approval behaviour for the documented HMDA cohort. It
does not predict default risk or creditworthiness, and its outputs are not
causal, legal, or policy evidence of discrimination. Raw data, frozen
artifacts, checkpoints, predictions, thresholds, bootstrap indices, manifests,
receipts, and accepted reports must remain unchanged unless a separate task
explicitly authorizes a new experiment.
