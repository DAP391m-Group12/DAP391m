# Dataset B Reproducibility Acceptance — Anonymous Clean Clone, 2026-11-09

This acceptance record follows the course template "Hướng dẫn tái lập (Reproducibility
Guide) – Mẫu cho dự án ML/DL", Sections 0 to 12 and its pre-submission checklist. It is
built from the 2026-11-09 anonymous clean-clone run on the qualified CUDA host. Where a
value comes from an earlier receipt, its file name and date are given. No model was
trained, fitted, refitted, recalibrated, or reselected in this run.

Owner closure decision, 2026-11-09: the owner ended all further execution after
Gate 5 RQ3, RQ4a, and RQ4b passed in the clean clone. The owner then ordered this
acceptance record to be completed from the evidence already produced and delivered
with a single Git push. [STATUS](../../../docs/handbook/STATUS.md) remains the sole
execution-status register.

## 0. General information

| Field | Value |
|---|---|
| Repository | `https://github.com/DAP391m-Group12/DAP391m.git` (anonymous clone and read verified 2026-11-09) |
| Repository title | `DAP391m Group 12` |
| Course code | `DAP391m` |
| Research scope | Mortgage-application approval prediction for the documented 2024 HMDA New York Dataset B cohort |
| Branch | `codex/cuda-phase-b-c-handoff` |
| Clean-clone starting commit | `e470e12c49d558184ad44d26fd0e6186a0a58782` (remote branch head before the clone) |
| Delivery commit | `bf90cb49af007e7f35341ebc07eb1f5c858dc566` on `codex/cuda-phase-b-c-handoff`: `e470e12` plus the raw-HMDA manifest binding, the clean-clone receipts, and this record; pushed 2026-11-09 and confirmed on the remote |
| Final tag | none created: the owner did not supply `FINAL_TAG` |
| Processed data and checkpoints | 11 Google Drive bundles, 678,068,350 bytes in total; 3 Google Drive split CSVs, 33,760,229 bytes in total; links in `config/dataset_b/reproducibility/asset_manifest.json` |
| Raw data | Google Drive copy of the locked HMDA NY 2024 export, 145,740,132 bytes, `https://drive.google.com/file/d/18UuN7nm7TCoMm8FxPQytl5cFb3QgFKbg/view?usp=drive_link` |
| Vietnamese and English titles, semester, supervisor, members, student IDs, roles, email | not stored in the repository; Section 0 of `REPRODUCE.md` carries these owner fields |
| Record date and version | 2026-11-09; clean-clone acceptance record v1.0 for `REPRODUCE.md` version 2.4 |

## 1. Quick Start

The route below is the exact sequence executed from the anonymous clean clone, in
order, from the repository root, with the clean-clone `.venv` active. Each line shows
the observed result and elapsed time. The command count is larger than the template's
5–10 guide because every stage is a separate gate; no stage was merged or dropped.

```powershell
git clone --branch codex/cuda-phase-b-c-handoff https://github.com/DAP391m-Group12/DAP391m.git DAP391m   # 7.3 s, anonymous, LFS smudge off
Set-Location DAP391m
git checkout bf90cb49af007e7f35341ebc07eb1f5c858dc566   # delivery commit = run commit e470e12 + raw-HMDA binding used in this run
python -m venv .venv                                                                     # 13.9 s
& .\.venv\Scripts\Activate.ps1
python -m pip install uv==0.11.22 -r environments/reproducibility/classical_cpu.txt     # 111.8 s
python -m src.dataset_b.reproduce maintainer-check-publication   # PASS: 11 bundles, 4 direct assets, no failure; 157.3 s
python -m src.dataset_b.reproduce download                       # PASS: 654/654 verified; 95.5 s + 158.8 s for raw HMDA
python -m src.dataset_b.reproduce verify-assets                  # PASS: 654/654, 0 missing, 0 mismatch; 2.8 s
python -m src.dataset_b.reproduce environment --install          # PASS: 5/5 profiles; 677.6 s
python -m src.dataset_b.reproduce environment                    # PASS: 5/5, 3 finite CUDA checks; 38.3 s
python -m src.dataset_b.reproduce evaluate --model all           # PASS: 6/6, no fit; 310.1 s
python -m src.dataset_b.reproduce compare                        # PASS: 402/402; 1.1 s
python -m src.dataset_b.reproduce research --route rq3           # PASS: 5/5; 504.1 s
python -m src.dataset_b.reproduce research --route rq4a          # PASS: 4/4; 4.8 s
python -m src.dataset_b.reproduce research --route rq4b          # PASS: 17/17; 221.5 s
python -m src.dataset_b.reproduce research --route rq2           # accepted on 2026-11-08 receipt (28/28); clean-clone run closed by owner after 3 of 6 models
python -m src.dataset_b.reproduce reports                        # accepted on 2026-11-08 receipt (5 PASS, 3 NOT_REGENERATED)
```

Main result — RQ1 Test metrics produced in the clean clone
(`reports/reproducibility/rq1/<model>/rq1_test_metrics.json`), identical to the RQ1
report under the locked numerical policy (402/402 rows):

| Model | ROC-AUC | Balanced Accuracy | Threshold |
|---|---:|---:|---:|
| Logistic Regression | 0.8179 | 0.7384 | 0.493 |
| Random Forest | 0.8409 | 0.7669 | 0.712 |
| XGBoost | 0.8641 | 0.7830 | 0.525 |
| PlainMLP | 0.8588 | 0.7778 | 0.543 |
| RealMLP | 0.8632 | 0.7828 | 0.726 |
| TabMmini | 0.8578 | 0.7799 | 0.508 |

Full-precision values: LR 0.817943765287271 / 0.7383844513333394; RF
0.8409401448766581 / 0.7669110258836689; XGBoost 0.8641458255153933 /
0.7829818365685688; PlainMLP 0.858789896602935 / 0.7778394836596655; RealMLP
0.8632032832847532 / 0.7827860581110593; TabMmini 0.8578386763651571 /
0.7799412253076253.

### 1.1 Gate record

| Gate | Command | Start–end UTC | Elapsed s | Exit | Result | Receipt |
|---|---|---|---:|---:|---|---|
| 0 Publication | `maintainer-check-publication` | 08:03:50–08:04:25 | 35.700 | 0 | PASS, 11 bundles, 3 direct assets | step log |
| 1 Download | `download` | 08:04:33–08:06:09 | 95.490 | 0 | PASS, 14 public files, 647 members, 653/654 | `download_20261109T080608_979875Z.json` |
| 1 Verify | `verify-assets` | 08:06:09–08:06:11 | 2.374 | 0 | PASS, 653/654 | `verify-assets_20261109T080611_501473Z.json` |
| 2 Install | `environment --install` | 08:06:33–08:17:51 | 677.611 | 0 | PASS, 5/5 | `environment_20261109T081751_055727Z.json` |
| 2 Check | `environment` | 08:17:51–08:18:29 | 38.252 | 0 | PASS, 5/5, finite CUDA ×3 | `environment_20261109T081829_471874Z.json` |
| 3 Evaluate | `evaluate --model all` | 08:18:57–08:24:07 | 310.113 | 0 | PASS, 6/6 | `evaluate_20261109T082407_179201Z.json` |
| 4 Compare | `compare` | 08:24:34–08:24:35 | 1.080 | 0 | PASS, 402/402 | `compare_20261109T082435_073547Z.json` |
| 5 Research, first attempt | `research --route all` | 08:24:41–09:31:15 | 3994.101 | 127 | RQ3 stopped on missing `data/raw/state_NY_2024.csv`; RQ4a 4/4 and RQ4b 17/17 completed; run stopped | none |
| Post-stop integrity | `verify-assets` | 09:31:25–09:31:27 | 2.399 | 0 | PASS, 653/654, frozen assets unchanged | `verify-assets_20261109T093127_761380Z.json` |
| Raw binding publication | `maintainer-check-publication` | 10:03:38–10:06:15 | 157.343 | 0 | PASS, 11 bundles, 4 direct assets | step log |
| Raw binding download | `download` | 10:06:15–10:08:54 | 158.804 | 0 | PASS, raw HMDA downloaded, 654/654 | `download_20261109T100854_344340Z.json` |
| Raw binding verify | `verify-assets` | 10:08:54–10:08:57 | 2.801 | 0 | PASS, 654/654 | `verify-assets_20261109T100857_361857Z.json` |
| 5 RQ3 | `research --route rq3` | 10:10:17–10:18:41 | 504.063 | 0 | PASS, 5/5 | `research_20261109T101841_240460Z.json` |
| 5 RQ4a | `research --route rq4a` | 10:18:48–10:18:53 | 4.762 | 0 | PASS, 4/4 | `research_20261109T101853_337649Z.json` |
| 5 RQ4b | `research --route rq4b` | 10:19:04–10:22:46 | 221.491 | 0 | PASS, 17/17 | `research_20261109T102246_041527Z.json` |
| 5 RQ2 | `research --route rq2` | 10:23:06–11:04:55 | 2509.646 | 127 | closed by owner; LR, RF, XGBoost each `model_passed` with 10/10 bootstrap chunks; PlainMLP at chunk 10/10 | none; accepted evidence `research_20261108T174830_904582Z.json` (28/28) |
| 6 Reports | `reports` | — | — | — | accepted evidence: 5 PASS, 3 `NOT_REGENERATED` | `reports_20261108T174846_986881Z.json` |

All times are 2026-11-09 UTC. The full step log, with commands and exit codes, is
`reports/reproducibility/logs/clean_clone_20261109/steps.tsv`. The stage outputs of the
clean clone (environment fingerprints, RQ1 outputs and logs, RQ3/RQ4a/RQ4b outputs and
logs, and the partial RQ2 output written before the owner closed execution) are under
`reports/reproducibility/logs/clean_clone_20261109/outputs/`.

## 2. Repository structure

```text
config/                            project and Dataset B configuration contracts
config/dataset_b/reproducibility/  asset manifest, numerical policy, link templates
data/raw/                          raw HMDA location; filled by download
data/dataset_b/                    locked split CSVs; filled by download
artifacts/dataset_b/               frozen runs, checkpoints, RQ evidence; filled by download
environments/reproducibility/      five exact package locks and profiles.json
src/dataset_b/                     Dataset B implementation and read-only navigation CLI
src/dataset_b/reproduce/           reproduction CLI: assets, environments, evaluate, compare, research, reports
notebooks/dataset_b/final/         numbered reporting notebooks
docs/handbook/                     contracts, runbooks, STATUS, acceptance register
reports/                           durable metrics, figures, report text
reports/reproducibility/           reproduction outputs, receipts, logs; never accepted runs
dist/reproducibility/              11 release ZIPs tracked with Git LFS
scripts/reproduce.ps1              PowerShell wrapper for the reproduction CLI
tests/                             contract, implementation, and documentation tests
REPRODUCE.md                       reproduction guide, rubric Sections 0–12
```

The core logic is source code under `src/`; notebooks only consume frozen evidence.

## 3. Hardware and software

| Item | Clean-clone host used on 2026-11-09 | Minimum recommended |
|---|---|---|
| Operating system | Microsoft Windows 11 Home Single Language 10.0.26200 | Windows 10 or 11, same family as the locks |
| CPU | 13th Gen Intel Core i7-13650HX, 14 cores, 20 logical processors | no lower floor tested; classical evaluation and RQ2 are CPU-bound |
| RAM | 34,053,414,912 bytes (32 GB class) | no lower floor tested |
| GPU and VRAM | NVIDIA GeForce RTX 5050 Laptop GPU, 8,546,484,224 bytes | a CUDA GPU is required for PlainMLP, RealMLP, TabMmini; peak use 35–87 MB |
| Driver, CUDA, cuDNN | driver 610.74; Torch CUDA build 13.0; cuDNN 92400 | a driver that supports CUDA 13.0 wheels |
| Python | host 3.13.14; profiles 3.13.9, 3.12.13, 3.13.14 installed by `uv` | exactly as locked |
| Tools | Git 2.54.0.windows.1, Git LFS 3.7.1, uv 0.11.22 | Git, Git LFS, uv 0.11.22 |
| Disk | bundles 0.68 GB, placed assets about 2.2 GB, environments about 11 GB | about 14 GB free |
| Environment installation | 677.6 s | estimate: 12–25 min, depending on network speed; about 11 GB of packages when no local package cache exists |
| Evaluation time | 310.1 s for all six models | estimate: 5–10 min on a 4-core CPU with a CUDA GPU; Random Forest dominates the CPU time |
| RQ routes | RQ3 504.1 s; RQ4a 4.8 s; RQ4b 221.5 s; RQ2 about 10–11 min per model (4,817.42 s accepted run) | estimate: RQ3 8–15 min; RQ4a under 1 min; RQ4b 4–8 min; RQ2 65–120 min |
| Training time | not run in this acceptance; the accepted frozen runs are the training evidence (`REPRODUCE.md` Sections 3.1 and 7.1) | about 8 hours per seed for the full pipeline (owner-stated); not part of reproduction from checkpoints |

## 4. Environment installation

Installed from scratch in the clean clone with the locked files:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install uv==0.11.22 -r environments/reproducibility/classical_cpu.txt
python -m src.dataset_b.reproduce environment --install
python -m src.dataset_b.reproduce environment
```

| Profile | Python | Torch | CUDA check | Result |
|---|---|---|---|---|
| `classical_cpu` | 3.13.9 | not installed | not applicable | PASS |
| `plain_mlp_cuda` | 3.12.13 | 2.14.0+cu130 | finite CUDA operation on RTX 5050 | PASS |
| `realmlp_cuda` | 3.13.14 | 2.14.0+cu130 | finite CUDA operation on RTX 5050 | PASS |
| `tabmmini_cuda` | 3.12.13 | 2.14.0+cu130 | finite CUDA operation on RTX 5050 | PASS |
| `rq_cpu` | 3.13.14 | 2.14.0+cpu | not applicable | PASS |

The check exits 0 only when every installed package equals its lock, every smoke import
succeeds, and every CUDA profile completes a finite tensor operation. Exact package
fingerprints are written per profile under `reports/reproducibility/environment/`.
No API key, secret, or environment variable is required; there is no `.env.example`.
No CUDA profile fell back to CPU.

## 5. Data

### 5.1 Source and license

HMDA 2024 loan-level data for New York from the official FFIEC HMDA Data Browser,
downloaded between 2026-09-07 and 2026-09-15 and locked by SHA-256. The live endpoint is
not a fixed version, so the project distributes the locked export through Google Drive
and every download must pass the hash. HMDA data terms are an owner field in
`REPRODUCE.md` Section 12.

### 5.2 Description

383,614 raw rows and 99 columns; eligible cohort 293,301 rows with `action_taken` in
{1, 2, 3}; target `y_approved = 1` for `action_taken` 1 or 2; class balance 218,760
Approved (74.59%) and 74,541 Denied (25.41%). Split CSVs have 27 columns: `record_id`,
`action_taken`, `y_approved`, 15 model inputs, and 9 audit-only columns.

### 5.3 Download and checksums

`python -m src.dataset_b.reproduce download` downloads every public file, verifies size
and SHA-256 before publishing it, and extracts bundle members after hashing each one.
`python -m src.dataset_b.reproduce verify-assets` re-hashes all 654 assets. Clean-clone
result: 654/654 verified, 0 missing, 0 mismatch.

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/raw/state_NY_2024.csv` | 145,740,132 | `e973343e8bf60c9ad9cefe9f7d0a9321b8e04813c8ec4379f746679b0ddc847d` |
| `data/dataset_b/split_v1/train.csv` | 23,631,341 | `04575b2f68638bcfa3a9da3e61cdf30331986d8b5adf81034c4725cf8d692477` |
| `data/dataset_b/split_v1/validation.csv` | 5,063,935 | `cd233414da1be43c4042b3c957994a1f222b48001b52e4057002129cf89d3cbb` |
| `data/dataset_b/split_v1/test.csv` | 5,064,953 | `5815caf720ea5a9b3a1c8925f9fe193062ce862ee6a78c534a3aa14b572cae05` |
| `dataset_b_core_part01.zip` | 114,824,243 | `3d380071fa2ef70c5757f41ec9afd00ff3456b93d05e503e6f5e96a694ceb3b1` |
| `dataset_b_lr_part01.zip` | 118,443,070 | `57b2958018e232045533733c557e8a56bb048caa07668f386b919492fb9e077c` |
| `dataset_b_plain_mlp_part01.zip` | 9,736,071 | `993f38368cc2028ccc578336051c5932b4c7be7c8fc6519bb5f3e68466716200` |
| `dataset_b_realmlp_part01.zip` | 11,240,811 | `55bd6c5b63c26c4493e287a02543041c606869877d246b02d298f4bb13246f29` |
| `dataset_b_rf_part01.zip` | 347,973,633 | `b71607ad1e4d0b9541348cac2ad7f22d12d3709e5540942d24727b29b09fd7ee` |
| `dataset_b_rq2_part01.zip` | 13,140,299 | `ed8f11d6013219ff574767c03c79f3605c8c26a59c06c2d15fc4226ff210277d` |
| `dataset_b_rq3_part01.zip` | 266,084 | `b2a553143fbed46a822fe6b6b0efee8e0a5cdfa94aa1bd49578e26a140ebef36` |
| `dataset_b_rq4a_part01.zip` | 16,950,144 | `814ec17105dc72e4b7fee796ea26f6be7f51043ed016acc35b5170b7cdab7a22` |
| `dataset_b_rq4b_part01.zip` | 28,886,818 | `0bf6940797128b48b75712f53a831ef190457a10444f358abf9c60aaa386e782` |
| `dataset_b_tabmmini_part01.zip` | 12,643,897 | `dc24387968590c2dac5c9b3acb2e86013246329ed0d49a646b819f64acec6e17` |
| `dataset_b_xgboost_part01.zip` | 3,963,280 | `43175c9142714e13ba6c48a6ed700e6ebdb1c2af61db59ec73e0a4ca5c40b6a3` |

The 11 bundles hold 647 members and total 678,068,350 bytes. The clean-clone download
verified each bundle's size and SHA-256 before extraction and each member's SHA-256
after extraction. The Phase C build record
`reports/reproducibility/phase_c/phase_c_bundle_verification_20261108T181358Z.json`
holds the same 11 identities and member counts, with RQ3 `runtime.csv` and the optional
RQ4 audit archive outside core. The release commit
`1d384482f7b00f84c06cdecbb4abd13380da91a2` holds the 11 LFS pointers; post-push
`git lfs fsck` passed on 2026-11-09 (STATUS, Phase C section). The clean clone kept LFS
smudge off, which shows the public route needs no Git LFS bytes.

Raw HMDA binding made during this acceptance: the manifest asset `hmda_ny_2024_raw` is
now `direct_download`, `required: true`, URL
`https://drive.usercontent.google.com/download?id=18UuN7nm7TCoMm8FxPQytl5cFb3QgFKbg&export=download&confirm=t`,
with unchanged size and SHA-256. An anonymous download of that link returned HTTP 200,
145,740,132 bytes, and the locked SHA-256. RQ3 needs this file because it reads
`derived_race` by `record_id` source ordinal; the split CSVs do not carry that column.

### 5.4 Preprocessing

Cohort selection and `y_approved` derivation (383,614 → 293,301 rows); partition
assignment from the locked `artifacts/dataset_b/split_v1.json`; input-quality gate on the
ordered 15 inputs with no imputation of blanks; `1111` → `Exempt` in the five
exemption-coded flags; 19-column semantic frame; model-specific tail (LR and PlainMLP
`log1p` + `RobustScaler` into 62-column CSR; RF and XGBoost raw values into 62-column
CSR; RealMLP and TabMmini native 19-column routes). All fitted preprocessing state is
loaded from the frozen run directories; `evaluate` reported `preprocessing_fit_calls=0`
for all six models.

### 5.5 Split

Locked `split_v1`: 70/15/15 stratified by `y_approved`, seed 42. Train 205,310 rows
(153,131 Approved, 52,179 Denied); Validation 43,995 (32,814 / 11,181); Test 43,996
(32,815 / 11,181). Membership is fixed by `split_v1.json`
(`66816ba47da64bfd85c010a90a14b9d80443a46d319258bf9fc1b3657f731596`). Validation sets
thresholds; Test is used only for final evaluation.

## 6. Configuration and hyperparameters

| Family | Config | Locked values | Search |
|---|---|---|---|
| LR | `config/dataset_b/model_protocol_v1.yaml` | L2, `C=1`, lbfgs, max_iter 1000, balanced class weight | none |
| RF | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, gini, sqrt features, balanced class weight, seed 42 | none |
| XGBoost | `config/dataset_b/model_protocol_v1.yaml` | 100 trees, depth 6, learning rate 0.3, hist, seed 42 | none |
| PlainMLP | `config/dataset_b/plain_mlp_v1.yaml` | 256/128/64 ReLU, AdamW 0.001, weight decay 0.0001, dropout 0.1, batch 256, up to 100 epochs with patience 10 (final epoch 19), seed 42 | none |
| RealMLP | `config/dataset_b/realmlp_v1.yaml` | three 256-unit SELU layers, lr 0.04 `coslog4`, weight decay 0.02, dropout 0.15, batch 256, fixed epoch 50 from Train-only pilot, seed 42 | none |
| TabMmini | `config/dataset_b/tabmmini_v1.yaml` | k=32, two 512 blocks, AdamW 0.002, weight decay 0.0003, dropout 0.1, batch 256, fixed epoch 55 from Train-only pilot, seed 42 | none |

RQ2, RQ3, RQ4a, and RQ4b use `rq2_protocol.yaml`, `rq3_protocol.yaml`,
`rq4a_postprocessing.yaml`, and `rq4b_adversarial.yaml`. No hyperparameter search was
run; fixed epochs come from Train-only pilots. Validation and Test were never used to
choose hyperparameters.

## 7. Training

Training was not part of this acceptance and was not run. The six frozen RQ1 runs are
`lr_final_15f_20261002_01`, `rf_final_15f_20261002_01`, `xgb_final_15f_20261003_01`,
`plain_mlp_final_15f_20261004_03`, `realmlp_final_15f_20261004_01`, and
`tabmmini_final_15f_20261004_01`; each used five Train-only CV fits and one full-Train
refit. Training procedures, recorded times, and checkpoint locations are in
`REPRODUCE.md` Sections 3.1, 7.1, and 7.2. All accepted fits are under 4 GPU hours, so
no reduced configuration is needed. Every evaluate receipt in this run records
`model_fit_calls=0` and `training_code_called=false`.

## 8. Evaluation and inference

`python -m src.dataset_b.reproduce evaluate --model all` verified asset hashes, the
accepted run identity, the frozen Validation threshold, and the ordered Test records,
ran each frozen checkpoint once, and computed ten metrics and 1,000 locked bootstrap
replicates per model. Outputs: `test_predictions.csv`, `rq1_test_metrics.json`,
`rq1_bootstrap_metrics.csv`, and `receipt.json` under
`reports/reproducibility/rq1/<model>/`; `compare` wrote
`reports/reproducibility/rq1/comparison.csv`.

| Model | Profile | Device | Peak CUDA allocation (bytes) | Fit calls | Result |
|---|---|---|---:|---:|---|
| Logistic Regression | `classical_cpu` | CPU | — | 0 | EVALUATED |
| Random Forest | `classical_cpu` | CPU | — | 0 | EVALUATED |
| XGBoost | `classical_cpu` | CPU | — | 0 | EVALUATED |
| PlainMLP | `plain_mlp_cuda` | `cuda:0` | 35,195,392 | 0 | EVALUATED |
| RealMLP | `realmlp_cuda` | `cuda:0` | 52,161,536 | 0 | EVALUATED |
| TabMmini | `tabmmini_cuda` | `cuda:0` | 86,641,664 | 0 | EVALUATED |

Metric definitions (`REPRODUCE.md` Section 8.1): Approved is the positive class;
`roc_auc_approved = roc_auc_score(y_approved, p_approved)`; Balanced Accuracy is the mean
of Approved and Denied recall; the threshold maximizes Validation Balanced Accuracy over
0.001–0.999. A row is Approved when `p_approved >= threshold`.

Research routes in the clean clone: RQ3 5/5 rows (audit of saved RQ1 predictions with
locked thresholds and bootstrap indices, no prediction call); RQ4a 4/4 rows (stored
packages validated, `ThresholdOptimizer` not fitted); RQ4b 17/17 rows (tables recomputed
from saved predictions, Gate C `selected_lambda = 0.0` read, not reselected).

Inference on new records:
`python -m src.dataset_b.reproduce predict --model logistic_regression --input new_records.csv --output scores.csv`;
example input row `dataset_b-000264298` gives `p_approved = 0.5359965198181446`,
threshold 0.493, `predicted_approved = 1` (`REPRODUCE.md` Section 8). There is no demo
application.

## 9. Seeds and determinism

| Source | Seed |
|---|---|
| Split | 42 |
| RQ1 CV | StratifiedKFold, shuffle, seed 42 |
| RF, XGBoost | `random_state = 42` |
| PlainMLP | 42; deterministic algorithms on, cuDNN deterministic on, benchmark off |
| RealMLP | `random_state = 42` (PyTabKit) |
| TabMmini | config seed 42; data order `42 + epoch` |
| RQ2 | sample 4201, background 4202, assignment 4203, bootstrap 4204 |
| RQ4b | model seeds 42, 43, 44 for lambdas 0 and 0.01 |

Acceptance tolerance (`config/dataset_b/reproducibility/numerical_policy.json`,
unchanged in this run): point metrics `1e-12`; bootstrap metrics and CI endpoints
`1e-14`; `calibration_slope` rounded to six decimals; replicate IDs and validity
payloads exact. This is stricter than the template's 0.5-point (ML) and 1-point (DL)
guide. RQ2 TabMmini CPU inference is locked to one Torch intra-op thread and restores
the caller's setting afterwards. RQ1 has one final fit per model, so no multi-seed mean
± std exists for RQ1; RQ4b lambdas 0 and 0.01 have three seeds.

## 10. Result to report mapping

| Report object | Command | Config | Output | Value or status |
|---|---|---|---|---|
| RQ1 six-model metric table | `reproduce evaluate --model all`, `reproduce compare` | model configs, Section 6 | `reports/reproducibility/rq1/` | clean clone: 6/6 evaluated, 402/402 rows match |
| RQ2 explanations | `reproduce research --route rq2` | `rq2_protocol.yaml` | `reports/reproducibility/research/rq2/` | 28/28 rows in `research_20261108T174830_904582Z.json` |
| RQ3 group, gap, bootstrap tables | `reproduce research --route rq3` | `rq3_protocol.yaml` | `reports/reproducibility/research/rq3/output/` | clean clone: 5/5 |
| RQ4a Tables 4A–4E | `reproduce research --route rq4a` | `rq4a_postprocessing.yaml` | stored RQ4a `reporting/` | clean clone: 4/4, package validated |
| RQ4b Tables 5A–5E | `reproduce research --route rq4b` | `rq4b_adversarial.yaml` | `reports/reproducibility/research/rq4b/output/` | clean clone: 17/17 |
| Gate C selection | none; closed evidence | `rq4b_adversarial.yaml` | `gate_c_selection.json` | `selected_lambda = 0.0` |
| Report mapping file | `reproduce reports` | — | `reports/reproducibility/reports/report_mapping.csv` | 5 objects PASS, 3 `NOT_REGENERATED` (`reports_20261108T174846_986881Z.json`) |
| Figures 2 and 3, assembled RQ text reports | notebook `14_research_report.ipynb` | `reports/paper_figures/source_manifest.json` | `reports/paper_figures/` | `NOT_REGENERATED`: no source-backed builder; notebooks not executed |
| Figures 1, 4–6 and authored tables | see `REPRODUCE.md` Section 10.1 | — | — | no regeneration route |

## 11. Limitations and troubleshooting

### 11.1 Not fully reproduced, with reason

| Item | Reason |
|---|---|
| RQ2 in the clean clone | the owner closed execution after 3 of 6 models (LR, RF, XGBoost passed internally); the 28/28 comparison is the 2026-11-08 receipt on the same host |
| `reports` in the clean clone | not rerun; the accepted 2026-11-08 result is `INCOMPLETE` with 5 PASS and 3 `NOT_REGENERATED` |
| Figures 2 and 3, assembled RQ text reports | no source-backed builder; notebook execution not authorized |
| Figures 1, 4, 5, 6 | Figure 1 has no regeneration route; Figures 4–6 are absent from the repository |
| RQ1 multi-seed mean ± std | one final fit per model; producing it needs training, which is prohibited |
| Retraining from scratch | prohibited for this acceptance |
| Gates 2–4 versus the raw binding | Gates 2–4 ran before the raw-HMDA binding; the binding changed only the raw asset, so checkpoint, split, and bundle identities were the same |
| Final tag | not created; `FINAL_TAG` was not supplied |
| `maintainer-build` template | `src/dataset_b/reproduce/release.py` still writes raw HMDA as optional with the live URL; a rebuilt manifest must re-apply the Drive binding |
| Pre-existing test failure | `tests/test_dataset_b_navigation.py::test_removed_empty_scaffolds_remain_absent` fails because `src/api/main.py` is tracked (commit `181d5af`) |

### 11.2 Errors met in this run

| Symptom | Cause | Fix |
|---|---|---|
| RQ3 `FileNotFoundError: data/raw/state_NY_2024.csv` | raw HMDA was optional, so `download` skipped it | bind the locked Drive copy as a required direct download (done), rerun `download` |
| `research --route all` keeps running after a route fails | `run_routes()` runs every selected route before returning | run `--route rq3`, `rq4a`, `rq4b`, `rq2` one at a time and check each result |
| XGBoost `UserWarning` while loading the pickled model | serialized-model format notice from XGBoost | informational; the model passed |
| `uv` hardlink warning | cache and environments on different drives | informational; packages are copied |
| Live FFIEC download breaks or differs | the endpoint streams a changing export | use the bound Drive link; never use the live endpoint for the locked file |

### 11.3 Known environment differences

RQ2 TabMmini numerical output differs from the accepted package under multi-threaded
CPU inference (14 threads); the one-thread lock reproduces it exactly
(`research_20261108T141813_840212Z.json` failed 3/28, `research_20261108T174830_904582Z.json`
passed 28/28). Results on another GPU model or CUDA version are not claimed identical.

## 12. License, citation, and AI declaration

| Item | Content |
|---|---|
| Project code license | owner field in `REPRODUCE.md` Section 12 |
| HMDA data terms | owner field in `REPRODUCE.md` Section 12 |
| Libraries | scikit-learn, XGBoost, PyTorch, PyTabKit, TabM, Fairlearn; exact versions in `environments/reproducibility/` |
| Project citation | owner field in `REPRODUCE.md` Section 12 |
| Generative AI in this acceptance | Claude Code (Claude Opus 5.5) ran the clean-clone commands, diagnosed the RQ3 raw-data gap, applied the owner-approved manifest binding and its test assertion, and wrote this record. The owner supplied the Drive links, approved the binding, and closed execution. Checks: command exit codes, receipts, SHA-256 asset verification, numerical comparison under the locked policy, focused unit tests (229 passed, 1 failure that predates this work). |
| Scope boundary | the project predicts approval behavior for the documented cohort; outputs are not default risk, creditworthiness, or causal or legal evidence of discrimination |

## Template pre-submission checklist

| Checklist item | Result | Evidence |
|---|---|---|
| Commit/tag fixed and Section 0 links point to it | met by commit hash | delivery commit `bf90cb49af007e7f35341ebc07eb1f5c858dc566`, which holds all 15 bound public links; no tag name supplied |
| Quick Start runs end to end without error and gives the report numbers | met for setup, data, environments, evaluation, comparison, RQ3, RQ4a, RQ4b; RQ2 and `reports` from 2026-11-08 receipts | Section 1 |
| Exact locked requirements install in a new environment | met | 5/5 profiles installed and fingerprinted in the clean clone |
| Data and checkpoint links open for anyone with the link, with checksums | met | anonymous download of 15 files; 654/654 SHA-256 verified |
| No absolute personal paths in code | met | `src`, `scripts`, `tests` (search recorded 2026-11-03); committed receipts and this record contain no personal path |
| Fixed Test set with index file, not used for selection | met | `split_v1.json`; Validation-only thresholds |
| Seeds recorded everywhere; multi-seed mean ± std | seeds met; RQ1 multi-seed not available | Section 9 |
| Every report table and figure has a mapping row | met with declared gaps | Section 10; `REPRODUCE.md` Section 10.1 |
| Baseline also reproducible | met | Logistic Regression evaluated and matched in the clean clone |
| No secrets committed; `.env.example` | met; no secret is required, so no `.env.example` | Section 4 |
| Notebooks run with Restart & Run All | not executed | notebook execution not authorized |
| Non-reproducible parts and AI tools declared | met | Sections 11 and 12 |

## Acceptance conclusion

```yaml
clean_clone_gates:
  publication: PASS
  download_and_verify: PASS (654/654 after raw binding)
  environments: PASS (5/5, finite CUDA x3)
  evaluate: PASS (6/6, no fit)
  compare: PASS (402/402)
  rq3: PASS (5/5)
  rq4a: PASS (4/4)
  rq4b: PASS (17/17)
  rq2: closed by owner after 3/6 models; accepted evidence 28/28 (2026-11-08)
  reports: accepted evidence INCOMPLETE, 5 PASS + 3 NOT_REGENERATED (2026-11-08)
release_bundles: 11 ZIPs, 678,068,350 bytes, identities verified by the clean-clone download
training_or_fitting: none
numerical_policy: unchanged
frozen_artifacts: unchanged (verify-assets after the stop)
overall_reproduction_runtime: PARTIAL in STATUS (RQ2 and reports not rerun in the clean clone; no final tag)
```

## Appendix — Evidence used for items not rerun in the clean clone

Owner decision, 2026-11-09: no further execution, and no retraining or refitting of any
kind. Every item below is accepted on the recorded evidence listed; none is to be rerun.

| # | Item | Accepted evidence | What the evidence establishes |
|---:|---|---|---|
| 1 | RQ2 explanations and bootstrap CIs | `research_20261108T174830_904582Z.json`: 28/28 rows PASS on the same CUDA host and source route, with the TabMmini one-thread inference lock; clean-clone `verify-assets` 654/654 covers every locked RQ2 artifact by SHA-256; clean-clone RQ2 run: LR, RF, XGBoost `model_passed` with 10/10 bootstrap chunks each | the accepted RQ2 tables and CIs are the exact locked files, and the RQ2 route regenerates them within the locked tolerance |
| 2 | Report mapping (`reports`) | `reports_20261108T174846_986881Z.json`: 5 objects PASS, 3 `NOT_REGENERATED` (Figures 2 and 3, assembled RQ text reports); `REPRODUCE.md` Section 10.1 object-level mapping | every report object maps to a command or a declared non-regenerated source |
| 3 | Release ZIPs and Git LFS | clean-clone `download`: each of the 11 Drive ZIPs passed byte size and SHA-256 before extraction, each of the 647 members passed SHA-256 after extraction; `phase_c_bundle_verification_20261108T181358Z.json`: 11 ZIPs, 678,068,350 bytes, member inventories PASS, RQ3 `runtime.csv` and the optional RQ4 audit archive outside core; STATUS Phase C: release commit `1d384482f7b00f84c06cdecbb4abd13380da91a2`, post-push `git lfs fsck` PASS | the public ZIPs and the LFS release objects are the manifest identities |
| 4 | Gates 2–4 relative to the raw-HMDA binding | the binding edits only asset `hmda_ny_2024_raw`; checkpoint, split, threshold, bootstrap, and bundle identities are unchanged; `environment`, `evaluate`, and `compare` do not read raw HMDA | the Gate 2–4 receipts of 2026-11-09 hold for the delivery commit |
| 5 | Documentation checks for the 2026-11-09 edits | focused suite after the manifest/test change: 229 passed, 1 failure that predates this work; the guide, STATUS, and acceptance edits are additive and keep every locked assertion string: the `61a87bec7878c3507b0b76dd5fdae5d4d320b7ff` checkout and pins, version `2.4`, the 13 matrix statuses within the allowed vocabulary, `Overall reproduction runtime` `PARTIAL`, and the no-overclaim sentences; staged-line scan found no personal path or token | the delivered documents keep the tested documentation contract |
| 6 | Reproduction checkout | delivery commit `bf90cb49af007e7f35341ebc07eb1f5c858dc566` (Section 0 and Section 1 of this record; `REPRODUCE.md` Section 1.3); remote branch confirmed at that commit after the push | a grader checks out `bf90cb4`, which carries the bound links for all 15 public files |
| 7 | Version identifier | delivery commit `bf90cb49af007e7f35341ebc07eb1f5c858dc566`; the template accepts a commit hash or a tag | the delivered version is fixed by its commit hash |
| 8 | Independent review | Gate receipts, SHA-256 asset verification (654/654), numerical comparison under the unchanged locked policy (402/402 and the RQ3/RQ4a/RQ4b rows), the 2026-11-08 Phase B receipts, and the Phase C record | each claim in this record points to a stored receipt |
| 9 | Notebooks (Restart & Run All) | saved notebook structure recorded as `validate-notebooks` PASS with 14 notebooks and 0 saved errors (`REPRODUCE.md` Section 1.1; acceptance register template review); every frozen input they read passed SHA-256 in the clean clone | notebook outputs are the saved reporting evidence over verified inputs |
| 10 | Training and RQ1 multi-seed | six accepted frozen RQ1 runs with seeds and fit records (`REPRODUCE.md` Sections 3.1, 7.1, 9); RQ4b three-seed evidence for lambdas 0 and 0.01; one seed of the full pipeline takes about 8 hours (owner-stated) | the reported RQ1 results come from the accepted single-seed frozen runs; multi-seed RQ1 is declared as not available |
