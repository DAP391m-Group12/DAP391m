# Post-processing versus Adversarial In-processing for Racial Disparities in Mortgage Approval Models

**DAP391m · Group 12 · FPT University HCMC · FA26**

This repository contains the Dataset B implementation, experimental protocols, and retained evidence for a study of mortgage application approval using HMDA New York 2024 data.

The study benchmarks six classification models, examines their explanations and group disparities, and compares Equalized-Odds post-processing with adversarial in-processing.

The prediction target is the observed mortgage application approval outcome. Results do not establish borrower default risk, overall creditworthiness, or causal or legal evidence of discrimination.

## Research Scope

The repository uses the following research-question identifiers:

| Repository identifier | Focus |
| --- | --- |
| RQ1 | Predictive performance, calibration, and paired comparisons across six models |
| RQ2 | Feature explanations using the locked GSV and interventional Shapley–Lorenz adaptation protocol |
| RQ3 | Descriptive fairness audit across sex and race groups |
| RQ4a | Equalized-Odds post-processing of frozen XGBoost and RealMLP predictions |
| RQ4b | Adversarial in-processing with RealMLP |

The mitigation study is the central comparison reflected in the project title. In the manuscript, repository RQ4a and RQ4b correspond to RQ3a and RQ3b.

Dataset B is the active experimental branch. Dataset A is retained for bridge analysis and has a separate data and artifact boundary.

## Dataset and Prediction Target

The experimental source is the locked HMDA New York 2024 CSV documented in the Dataset B contracts.

Only records with `action_taken ∈ {1, 2, 3}` enter the eligible cohort:

| HMDA outcome | Target |
| --- | --- |
| `action_taken ∈ {1, 2}` | `y_approved = 1` |
| `action_taken = 3` | `y_approved = 0` |

The eligible cohort contains **293,301 records**, partitioned using the fixed, target-stratified `split_v1`:

| Partition | Records | Role |
| --- | ---: | --- |
| Train | 205,310 | Train-only cross-validation and full-Train fitting |
| Validation | 43,995 | Threshold selection and protocol-defined mitigation selection |
| Test | 43,996 | Final evaluation after the applicable decisions are locked |

The benchmark uses **15 model-input fields**. Technical identifiers, target fields, and audit-only metadata remain outside the main-model predictor matrix.

Race audits use the documented application-level `derived_race` aggregation. Protected-group definitions, population coverage, and denominator rules follow the corresponding fairness and mitigation contracts.

## Models and Evaluation

The six benchmark models are:

- Logistic Regression
- Random Forest
- XGBoost
- PlainMLP
- RealMLP
- TabMmini

The benchmark uses five-fold stratified cross-validation within Train. Learned preprocessing is fitted only on the permitted training population. Each final model is fitted on full Train, its classification threshold is selected on Validation, and Test is reserved for final reporting.

Evaluation covers discrimination, class-specific performance, calibration, and uncertainty. Cross-model and mitigation comparisons follow the locked paired-bootstrap policies; point-estimate differences alone do not establish statistical significance.

## Fairness Mitigation

**Post-processing — RQ4a**

Equalized-Odds post-processing consumes frozen XGBoost and RealMLP Validation/Test predictions. It does not retrain the base models.

**Adversarial in-processing — RQ4b**

The adversarial study trains separate RealMLP experimental models under the approved mitigation protocol. Race and sex remain excluded from the predictor inputs.

Both primary mitigation analyses use the fixed G3 race roster: White, Black or African American, and Asian. RQ4a also includes the required G6 sensitivity analysis. The optional RQ4b G6 sensitivity remains deferred.

Execution acceptance and scientific effectiveness are separate conclusions. Accepted packages do not, by themselves, establish that mitigation improves fairness or preserves predictive utility.

## Repository Structure

```text
DAP391m/
├── config/dataset_b/             # Dataset B configurations and protocols
├── data/
│   ├── raw/                      # Locked HMDA source
│   └── dataset_b/                # Derived views and fixed split CSVs
├── artifacts/dataset_b/          # Model, prediction, and research evidence
├── environments/reproducibility/ # Environment profiles and dependency locks
├── src/dataset_b/                # Dataset B implementation
│   └── reproduce/                # Reproduction command-line interface
├── notebooks/dataset_b/          # Analysis and reporting notebooks
├── reports/                      # Research reports and figures
├── docs/handbook/                # Contracts, runbooks, and execution register
├── scripts/                      # Project command wrappers
├── tests/                        # Relevant implementation and contract tests
└── REPRODUCE.md                  # Reproduction instructions
```

Data, checkpoints, and other large assets are distributed separately from the ordinary Git source files. Follow the reproduction guide to obtain and verify the required assets.

## Reproduction

Start with [REPRODUCE.md](REPRODUCE.md) for environment setup, asset download, checksum verification, frozen-checkpoint evaluation, and research-output comparison.

The reproduction route evaluates retained checkpoints and checks outputs against accepted evidence. Fresh model training is outside the Quick Start.

**Reproducibility limitation:** the current execution register records overall reproduction runtime as `PARTIAL`. Some stages have successful clean-clone evidence, but complete end-to-end reproduction from the current source has not been established. See [STATUS](docs/handbook/STATUS.md) for the evidence and remaining limitations.

## Documentation

| Purpose | Source |
| --- | --- |
| Find the applicable Dataset B documentation | [Handbook](docs/handbook/README.md) |
| Understand the cohort, target, schema, and leakage boundaries | [Data contract](docs/handbook/contracts/data_contract.md) |
| Understand feature selection and the final input contract | [Feature governance](docs/handbook/contracts/feature_governance.md) |
| Understand splits, CV, thresholds, metrics, and bootstrap comparisons | [Split and evaluation contract](docs/handbook/contracts/split_and_evaluation.md) |
| Understand explanation methodology | [RQ2 explanation contract](docs/handbook/contracts/rq2_explanation.md) |
| Understand descriptive fairness evaluation | [RQ3 fairness contract](docs/handbook/contracts/rq3_fairness.md) |
| Understand mitigation methods and comparison boundaries | [RQ4 mitigation plan](docs/handbook/RQ4_MITIGATION_PLAN.md) |
| Check accepted runs and current execution status | [STATUS](docs/handbook/STATUS.md) |
| Review documentation-level reproducibility acceptance | [Reproducibility acceptance](docs/handbook/REPRODUCIBILITY_ACCEPTANCE.md) |

`STATUS.md` is the sole current execution-status register. Contracts define methodology; runbooks define procedures; retained artifacts provide evidence.

## Data and Artifact Integrity

- Preserve the locked raw source and accepted experimental artifacts.
- Keep Dataset A and Dataset B fitted objects, splits, predictions, and results separate.
- Do not use Test to select features, fit preprocessing, tune models, or choose thresholds.
- Preserve record order, prediction semantics, fixed group rosters, and locked bootstrap indices.
- Do not overwrite accepted runs when performing a new verification.
