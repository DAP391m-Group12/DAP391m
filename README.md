# HMDA New York 2024 Mortgage Approval Study

This project studies historical mortgage approval decisions in HMDA New York 2024. The target is a binary historical decision, not creditworthiness or loan default:

- `y_approved = 1` when `action_taken` is `1` or `2`.
- `y_approved = 0` when `action_taken` is `3`.
- `action_taken` is used only to create the target and must never be a model input.
- Every model must produce the continuous probability `P(approved)`.

## Locked Set A scope

Set A is a bridge analysis for RQ1 and RQ2. It uses only Logistic Regression and Random Forest on one identical dataset split. It does not answer the full cross-model questions and must not be used to draw conclusions about XGBoost, plain MLP, RealMLP, or TabMmini; those belong to Set B.

Set A uses 11 raw inputs:

- `applicant_race-1` through `applicant_race-5`
- `applicant_sex`, `income`, `loan_amount`, `loan_purpose`, `lien_status`, `loan_type`

The five race slots must be retained and multi-hot encoded without treating their positions as ordered. Encoded race columns are grouped back to `applicant_race`; the seven canonical explanation groups are `applicant_race`, `applicant_sex`, `applicant_income`, `loan_amount`, `loan_purpose`, `lien_status`, and `loan_type`.

## Evaluation contract

Hyperparameters are tuned only by cross-validation within the training split. Each model selects its own validation threshold by maximizing balanced accuracy, then both the fitted model and threshold are frozen. The test split is used once for final assessment.

RQ1 reports ROC-AUC, Denied PR-AUC, Brier score, calibration intercept/slope/curve, balanced accuracy, precision/recall/F1 for both Approved and Denied, plus training and inference time. Probability calibration is evaluated but is not applied as a model-adjustment step.

RQ2 explains only `P(approved)` with four separate analyses: LR-GSV, LR-Shapley-Lorenz, RF-GSV, and RF-Shapley-Lorenz. Contributions are summed with sign within each explanation group before absolute-value L1 normalization; raw values from GSV and Shapley-Lorenz are never compared directly.

Both RQ1 and RQ2 use paired bootstrap with 1,000 resamples and percentile 95% confidence intervals. No model is retrained during bootstrap. RQ1 shares test-set bootstrap indices across models; RQ2 shares explanation-observation indices across explanation methods.

## Configuration

- `config/config.yaml` contains the study-wide data, split, and bootstrap contract.
- `config/features.yaml` defines the target, locked Set A features, and future Set B boundary.
- `config/model_config.yaml` defines the Set A model, threshold, evaluation, and explanation requirements.

## Running Set A

From the project root, install dependencies and execute the full reproducible Set A run:

```powershell
python -m pip install -r requirements.txt
python -m src.pipeline.train
```

The run loads only the locked Set A columns, creates one stratified 60/20/20
split with seed 42, tunes Logistic Regression and Random Forest using CV only
within the training split, freezes a model-specific validation threshold, and
prints the test-set comparison table. It writes frozen model artifacts to
`model/set_a/` and metrics/calibration/paired-bootstrap reports to
`reports/set_a/`.

For a quicker development run that skips only the 1,000-resample paired
bootstrap, use:

```powershell
python -m src.pipeline.train --skip-bootstrap
```

## Scope boundary

The project evaluates historical decisions and model predictions. It does not predict default, infer creditworthiness, or establish causal or legal conclusions about discrimination.
