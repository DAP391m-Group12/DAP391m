# Reproducibility Guide

## Contents

- [0. General Information](#0-general-information)
- [1. Reproducibility Summary (Quick Start)](#1-reproducibility-summary-quick-start)
- [2. Directory Structure](#2-directory-structure)
- [3. Hardware and Software Requirements](#3-hardware-and-software-requirements)
- [4. Environment Setup](#4-environment-setup)
- [5. Data](#5-data)
- [6. Configuration and Hyperparameters](#6-configuration-and-hyperparameters)
- [7. Training](#7-training)
- [8. Evaluation and Inference](#8-evaluation-and-inference)
- [9. Randomness Control and Determinism](#9-randomness-control-and-determinism)
- [10. Mapping Results ↔ Report](#10-mapping-results--report)
- [11. Limitations and Troubleshooting](#11-limitations-and-troubleshooting)
- [12. License, Citations, AI Disclosure](#12-license-citations-ai-disclosure)

## 0. General Information

| Field | Evidence-backed information / items to finalize |
| --- | --- |
| Vietnamese project title | Post-processing versus Adversarial In-processing for Racial Disparities in Mortgage Approval Models |
| English project title | Post-processing versus Adversarial In-processing for Racial Disparities in Mortgage Approval Models |
| Repository name/group | DAP391m - Group 12 |
| Course | DAP391m |
| Semester, Supervisor | FPTUHCM - FA26 |
| Members | Huỳnh Gia Bảo – SE204913<br>Nguyễn An Duy - SE203823<br>Lê Nam Phong - SE204029 |
| Repository | https://github.com/DAP391m-Group12/DAP391m.git |
| Version | PR #12 merge 369fe494506fce888d6c917f504d086a1dfcb097 |
| Data | https://drive.google.com/drive/folders/17Kk0_zJA-h018nFupMoD9ucPdz28eRgL?usp=drive_link |
| Checkpoint | https://drive.google.com/drive/folders/1ws2G8tBKOmH62jsOWS4DnJBa9lpE2NdE?usp=drive_link |
| Bundle size | 11 ZIP: 678,068,350 bytes in total |
| Source document version | 2.4 – 2026-11-09 |

## 1. Reproducibility Summary (Quick Start)

### 1.1. Clone the Repository

```powershell
git clone https://github.com/DAP391m-Group12/DAP391m.git
cd DAP391m
git checkout codex/cuda-phase-b-c-handoff
```

### 1.2. Set Up the Host Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install uv==0.11.22 -r environments/reproducibility/classical_cpu.txt
```

### 1.3. Install Profiles and Verify Assets

Create five profiles and obtain the exact data/checkpoints with checksums.

```powershell
python -m src.dataset_b.reproduce environment --install
python -m src.dataset_b.reproduce download
python -m src.dataset_b.reproduce verify-assets
```

### 1.4. Evaluate Frozen Checkpoints

Evaluate Test checkpoints and compare against the accepted evidence.

```powershell
python -m src.dataset_b.reproduce evaluate --model all
python -m src.dataset_b.reproduce compare
```

### 1.5. Expected RQ1 Test Results

| Model | ROC-AUC | Balanced Accuracy | Threshold |
| --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.8179 | 0.7384 | 0.493 |
| Random Forest | 0.8409 | 0.7669 | 0.712 |
| XGBoost | 0.8641 | 0.7830 | 0.525 |
| PlainMLP | 0.8588 | 0.7778 | 0.543 |
| RealMLP | 0.8632 | 0.7828 | 0.726 |
| TabMmini | 0.8578 | 0.7799 | 0.508 |

### 1.6. Additional Verification

```powershell
python -m src.dataset_b.reproduce environment
python -m src.dataset_b.reproduce research --route rq3
python -m src.dataset_b.reproduce research --route rq4a
python -m src.dataset_b.reproduce research --route rq4b
python -m src.dataset_b.reproduce research --route rq2
python -m src.dataset_b.reproduce reports
```

## 2. Directory Structure

```text
DAP391m/
├── config/dataset_b/              # catalog, model/RQ protocols, and reproducibility manifest
│   └── reproducibility/           # asset_manifest.json, numerical_policy.json, link examples
├── data/raw/                      # hash-locked raw HMDA; downloaded separately, not committed to Git
├── data/dataset_b/                # three fixed Train/Validation/Test split CSVs
├── artifacts/dataset_b/           # checkpoints, run manifests, predictions, bootstrap, RQ packages
├── environments/reproducibility/  # 5 lock files and profiles.json
├── src/dataset_b/                 # data processing, models, metrics code
│   └── reproduce/                 # CLI download / verify / environment / evaluate / research
├── notebooks/dataset_b/final/     # reporting notebooks and historical exploration
├── scripts/reproduce.ps1         # CLI wrapper for PowerShell
├── reports/                       # finalized reports and figures
│   └── reproducibility/           # results of new verification runs, not replacing accepted runs
├── docs/handbook/                 # contract, runbook, STATUS, acceptance
├── dist/reproducibility/          # release ZIPs; Git LFS on the handoff branch
├── tests/                         # code/contract tests
└── REPRODUCE.md                   # file required for submission at the root
```

## 3. Hardware and Software Requirements

| Component | Actual machine recorded in clean clone 2026-11-09 |
| --- | --- |
| OS | Windows 11, build 10.0.26200 |
| CPU | Intel Core i7-13650HX (14 cores/20 threads) |
| RAM | 34,053,414,912 bytes (~31.7 GiB) |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM | 8,546,484,224 bytes (nominal 8 GB) |
| NVIDIA driver / CUDA build / cuDNN | 610.74 / 13.0 / 92400 |
| PyTorch GPU | 2.14.0+cu130 |
| Python | 3.13.14 |
| Disk space | ZIP ~0.65 GB; assets after extraction ~2.2 GB; env ~11 GB |
| Time | Install ~10m, evaluate ~30-48h |

## 4. Environment Setup

| Profile | Python | Device | Lock file and purpose |
| --- | --- | --- | --- |
| classical_cpu | 3.13.14 | CPU | classical_cpu.txt: LR, RF, XGBoost |
| plain_mlp_cuda | 3.12.14 | CUDA | plain_mlp_cuda.txt: PlainMLP, torch 2.14.0+cu130 |
| realmlp_cuda | 3.13.14 | CUDA | realmlp_cuda.txt: RealMLP, torch 2.14.0+cu130 |
| tabmmini_cuda | 3.12.14 | CUDA | tabmmini_cuda.txt: TabMmini, torch 2.14.0+cu130 |
| rq_cpu | 3.13.14 | CPU | rq_cpu.txt: RQ2, RQ3, RQ4a, RQ4b |

After cloning, checking out, creating/activating .venv and installing the host lock as in Section 1:

```powershell
python -m src.dataset_b.reproduce environment --install
python -m src.dataset_b.reproduce environment
```

The latter step writes a receipt to reports/reproducibility/environment/

## 5. Data

### 5.1. Source and License

**Data source:** Home Mortgage Disclosure Act (HMDA) 2024 dataset, covering New York (NY) State, provided through the HMDA Data Browser.

**Official source:** https://ffiec.cfpb.gov/data-browser/data/2024?category=states

**Size:** 145,740,132 bytes.

**SHA-256:** `e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d`

Collection date: The data used for the runs was collected on 5/09/2026

Data download source for reproduction:

**Download link:** https://drive.google.com/file/d//view?usp=drive_link

**Data source:** Federal Financial Institutions Examination Council (FFIEC) / Consumer Financial Protection Bureau (CFPB), Home Mortgage Disclosure Act (HMDA) Data, 2024.

### 5.2. Dataset Overview

| Content | Recorded value |
| --- | --- |
| Raw | 383,614 data rows × 99 columns |
| Eligible cohort | 293,301 rows with action_taken ∈ {1,2,3}; 90,313 rows excluded |
| Label | y_approved=1 for action_taken ∈ {1,2}; 0 for action_taken=3 |
| Distribution | Approved: 218,760 (74.59%); Denied: 74,541 (25.41%) |
| Schema split CSV | 27 columns = record_id, action_taken, y_approved, 15 model inputs, 9 audit-only columns |
| Notable blank or NA indicators among the 15 inputs | debt_to_income_ratio 7.66%; income 7.02%; loan_term 3.53%; loan_to_value_ratio 3.40%; remaining 11 inputs 0% |

NA and Exempt can be valid under the data contract; do not treat all such values as blanks requiring imputation.

### 5.3. Download and Verify Checksums

```powershell
python -m src.dataset_b.reproduce download
python -m src.dataset_b.reproduce verify-assets
Get-FileHash -Algorithm SHA256 -LiteralPath data/raw/state_NY_2024.csv
Get-FileHash -Algorithm SHA256 -LiteralPath artifacts/dataset_b/split_v1.json
```

### 5.4. Preprocessing (in the Correct Order)

1. **src/dataset_b/data/dataset_view.py** – DatasetBViewBuilder: filter 383,614 → 293,301 rows and create y_approved.

2. **artifacts/dataset_b/split_v1.json** to write the exact three split CSVs; do not run **create_split.py** as a new split, especially not with --force (would replace the manifest).

3. **src/dataset_b/data/split_inputs.py** – DatasetBInputQualityGate: check 15 input fields against the contract. Real blanks are rejected, not replaced with MISSING; source checks show no contract blanks in the frozen splits.

4. **numeric_two_channel.py, median_imputer.py:** build a 19-column semantic frame (4 numeric-value channels, 4 status channels, 11 categorical variables); fit medians only on the permitted set, never on Validation/Test.

5. **Family-specific preprocessing:** LR and PlainMLP apply log1p to income_value, loan_amount_value, RobustScaler → 62-column CSR; RF/XGBoost keep raw numbers → 62-column CSR; RealMLP log1p → native 19-column frame; TabMmini log1p → 19 columns, category index.

### 5.5. Data Splitting and Leakage Prevention

The split is **stratified**, seed 42, **70/15/15** ratio; Validation/Test sets must not be used to fit the preprocessor or select hyperparameters; Validation selects the threshold, and Test is used only for final reporting.

| Split | Rows | Approved | Denied | Bytes | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| Train | 205,310 | 153,131 | 52,179 | 23,631,341 | `04575b2f68638bcfa3a9da3e61cdf30331986d8b5adf81034c4725cf8d692477` |
| Validation | 43,995 | 32,814 | 11,181 | 5,063,935 | `cd233414da1be43c4042b3c957994a1f222b48001b52e4057002129cf89d3cbb` |
| Test | 43,996 | 32,815 | 11,181 | 5,064,953 | `5815caf720ea5a9b3a1c8925f9fe193062ce862ee6a78c534a3aa14b572cae05` |

## 6. Configuration and Hyperparameters

Navigation catalog: config/dataset_b/catalog.yaml. Experimental/target RQ configurations are specific YAML files; do not use current YAML states to overwrite STATUS or historical manifests.

| Model | Configuration | Finalized hyperparameters / selection source |
| --- | --- | --- |
| Logistic Regression<br>(baseline) | config/dataset_b/model_protocol_v1.yaml | L2, C=1, lbfgs, max_iter=1000, class_weight=balanced; fixed, no HPO |
| Random Forest | config/dataset_b/model_protocol_v1.yaml | 100 trees, gini, max_features=sqrt, balanced, random_state=42; no HPO |
| XGBoost | config/dataset_b/model_protocol_v1.yaml | 100 trees, depth 6, learning rate 0.3, hist, seed=42; no HPO |
| PlainMLP | config/dataset_b/plain_mlp_v1.yaml | 256/128/64 ReLU, AdamW lr 0.001, wd 0.0001, dropout 0.1, batch 256, max 100 epochs, patience 10, best epoch 19; float64, AMP off |
| RealMLP | config/dataset_b/realmlp_v1.yaml | 3×256 SELU, Adam lr 0.04 (coslog4), wd 0.02, dropout 0.15 (flat_cos), batch 256, fixed epoch 50 from Train-only pilot; mixed precision NOT_RECORDED |
| TabMmini | config/dataset_b/tabmmini_v1.yaml | k=32, 2×512 blocks, AdamW lr 0.002, wd 0.0003, dropout 0.1, batch 256, fixed epoch 55 from Train-only pilot, amp=false |

## 7. Training

### 7.1. Runs Completed According to Historical Evidence

| Family | Accepted run ID | Seed / status |
| --- | --- | --- |
| LR | lr_final_15f_20261002_01 | split/CV seed 42; lbfgs has no model seed |
| RF | rf_final_15f_20261002_01 | split/CV seed 42; model seed 42 |
| XGBoost | xgb_final_15f_20261003_01 | split/CV seed 42; model seed 42 |
| PlainMLP | plain_mlp_final_15f_20261004_03 | seed 42; deterministic algorithm per config |
| RealMLP | realmlp_final_15f_20261004_01 | seed 42; fixed epoch 50 |
| TabMmini | tabmmini_final_15f_20261004_01 | config seed 42; fixed epoch 55 |

### 7.2. Main Checkpoints (Frozen Evidence, Not Part of the Git Source)

| Model | Path under artifacts/dataset_b/ | Bytes | SHA-256 |
| --- | --- | --- | --- |
| LR | `lr/final/lr_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib` | 10,099 | `69770bc11558f56f147e2af89f79747c903e74e779aeae17d1347a25de620859` |
| RF | `rf/final/rf_final_15f_20261002_01/checkpoints/fitted_pipeline.joblib` | 596,820,036 | `2e2378596ba62f4c82b7d657d45cb733223612fb38943a5e121a818183b6c802` |
| XGBoost | `xgboost/final/xgb_final_15f_20261003_01/checkpoints/fitted_pipeline.joblib` | 400,438 | `d24db9bb642e6e7810596e4a16632b1be88e68eeb030ad05130b134ea65ee834` |
| PlainMLP | `plain_mlp/final/plain_mlp_final_15f_20261004_03/full_train/checkpoints/fitted_bundle.joblib` | 446,219 | `bad3761eeb72f41b6cb2784530a1f46fce6b9bb2948c812f7d6e5e4337ae1968` |
| RealMLP | `realmlp/final/realmlp_final_15f_20261004_01/full_train/fitted_model.joblib` | 663,830 | `b5b79dcd3200f989ecd9cdae850fccaeb2f98ae0487046c68cebbdbcf8ea2754` |
| TabMmini | `tabmmini/final/tabmmini_final_15f_20261004_01/full_train/model/inference_bundle.pt` | 1,272,134 | `0d3eacadb7a2e07ae5a6fb7396f87b00317a21711a0b0db8b6aaf4ec1e4d2229` |

### 7.3. Training

| Family | Procedure if the owner authorizes it (not part of Quick Start) |
| --- | --- |
| LR | src.dataset_b.models.classical.lr.training |
| RF / XGBoost | src.dataset_b.models.classical.rf.training<br>src.dataset_b.models.classical.xgboost.training |
| PlainMLP | src.dataset_b.models.neural.plain_mlp.cv<br>src.dataset_b.models.neural.plain_mlp.training |
| RealMLP | src.dataset_b.models.neural.realmlp.cv<br>src.dataset_b.models.neural.realmlp.full_train |
| TabMmini | Src.dataset_b.models.neural.tabmmini.training.py |

## 8. Evaluation and Inference

### 8.1. Test Evaluation from Checkpoints

```powershell
python -m src.dataset_b.reproduce evaluate --model all
python -m src.dataset_b.reproduce compare
```

| Metric | Calculation / contract |
| --- | --- |
| roc_auc_approved | sklearn.metrics.roc_auc_score(y_approved, p_approved) |
| average_precision_denied | average_precision_score(1-y, 1-p) |
| precision_denied,<br>recall_denied, f1_denied | respectively TN/(TN+FN), TN/(TN+FP), 2TN/(2TN+FN+FP) |
| recall_approved | TP/(TP+FN) |
| balanced_accuracy | average of Approved/Denied recall |
| brier_score_approved | average (p_approved-y_approved)^2 |
| calibration_intercept,<br>calibration_slope | logistic calibration on logit(p) with clipping [1e-15,1-1e-15] |

### 8.2. Other RQ Branches

```powershell
python -m src.dataset_b.reproduce research --route rq2
python -m src.dataset_b.reproduce research --route rq3
python -m src.dataset_b.reproduce research --route rq4a
python -m src.dataset_b.reproduce research --route rq4b
```

## 9. Randomness Control and Determinism

| Source | Rule recorded in config/code | Limitations |
| --- | --- | --- |
| Data splitting | Stratified 70/15/15, seed 42, immutable split_v1.json | do not split again arbitrarily |
| CV RQ1 | 5-fold StratifiedKFold on Train, shuffle seed 42 | variation across folds differs from variation across seeds |
| LR | lbfgs, no estimator-specific seed | do not add a fictitious seed |
| RF/XGBoost | model random state/seed 42 | retain finalized parameters |
| PlainMLP | Python random, NumPy, Torch/CUDA seed 42;<br>deterministic algorithms & cuDNN deterministic on,<br>benchmark off | different GPU/CUDA may produce differences |
| RealMLP | random_state=42 passed to PyTabKit | project does not show an overall deterministic flag; internals have not been checked |
| TabMmini | config seed 42; data order generator 42 + epoch | strict determinism is not guaranteed |
| RQ2 | sample/background/assignment 4201/4202/4203, bootstrap 4204 | 1,000 replicates |
| RQ4b | 42/43/44 only for lambda 0 and 0.01; other screening only seed 42 | do not combine screening into multi-seed |

## 10. Mapping Results ↔ Report

| Table/figure/scope | Corresponding command / action | Config / run, result path |
| --- | --- | --- |
| RQ1: six-model Test<br>metrics, headline table | reproduce evaluate --model all | reports/RQ1_Research_Metrics_Final.txt |
| RQ2: explanation xAI | reproduce research --route rq2 | config/dataset_b/rq2_protocol.yaml;<br>run rq2_cpu_production_20261009_01;<br>reports/reproducibility/research/rq2/ |
| RQ3: group/gap/bootstrap | reproduce research --route rq3 | config/dataset_b/rq3_protocol.yaml;<br>rq3_fairness_20261010_01;<br>reports/reproducibility/research/rq3/output/ |
| RQ4a: Tables 4A–4E | reproduce research --route rq4a | config/dataset_b/rq4a_postprocessing.yaml;<br>runs rq4a3_postproc_20261017_01,<br>rq4a6_postproc_20261017_01 |
| RQ4b | reproduce research --route rq4b | config/dataset_b/rq4b_adversarial.yaml;<br>Test rq4b3_test_20261024_01;<br>reports/reproducibility/research/rq4b/output/<br>rq4b3_sex_supplementary_20261024_02,<br>closed development runs |
| RQ4b: | no builder with the same CLI | rq4b3_sex_supplementary_20261024_02,<br>closed development runs |
| Figure 1 | pipeline.png | reports/paper_figures/pipeline.png |
| Figures 2 and 3 | notebook 14_research_report.ipynb | reports/paper_figures/, source_manifest.json |
| Figures 4–6 | notebook 14_research_report.ipynb | figure_index.json; reports/paper_figures/ |
| Publication figures | notebook 10_publication_figures.ipynb | reports/figures/publication/ |

## 11. Limitations and Troubleshooting

### 11.1. Common Error Table

| Symptom | Possible cause / handling according to the source document |
| --- | --- |
| Notebook displays error output | Only the structure of the saved notebook has been checked; a separate testing task is needed if requested by the instructor |
| Dataset HMDA | Error due to dataset permissions and access |
| Backend configuration | Model backend configuration may vary between versions and a machine may be unable to run it |
| Runtime | Runtime is currently only an estimate; actual execution time may be much longer |

## 12. License, Citations, AI Disclosure

| Item | Status to disclose |
| --- | --- |
| Group source code license | MIT |
| HMDA data license | Federal Financial Institutions Examination Council (FFIEC) / Consumer Financial Protection Bureau (CFPB), Home Mortgage Disclosure Act (HMDA) Data, 2024. |
| Referenced libraries | scikit-learn,XGBoost,PyTorch,PyTabKit,TabM,Fairlearn |
