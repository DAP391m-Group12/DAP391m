# Dataset B RQ3 descriptive fairness audit contract

## 1. Purpose and status

RQ3 audits the six frozen final Dataset B models on the fixed Test partition.
It compares predicted loan-approval behavior across `applicant_sex` and
`derived_race`. The audit is descriptive. It does not retrain a model, rerun
inference, change a Validation threshold, mitigate a model, or support causal,
legal-discrimination, creditworthiness, or default-risk conclusions.

The machine-readable authority is `config/dataset_b/rq3_protocol.yaml`. The
protocol, implementation, tests, accepted production artifacts, and read-only
results notebook have passed their gates for run `rq3_fairness_20261010_01`.
Any later rerun or protocol change requires a separately authorized task.

## 2. Inputs and data boundary

RQ3 uses the six final prediction tables named in the protocol. Every table
must contain the same 43,996 ordered Test `record_id` values and the same
`y_approved`. The stored `p_approved` and each model's frozen Validation
threshold are the only model outputs used. A missing stored predicted class is
derived as `p_approved >= threshold`; this is not inference.

`applicant_sex` comes from the controlled Test view. `derived_race` is joined
from the immutable raw HMDA source through the one-based source-row number
encoded in the numeric suffix of `record_id`; dataframe position is therefore
`suffix - 1`. The join must be one-to-one and complete. Preflight must reconcile
`lei`, `action_taken`, and `applicant_sex` between every selected raw row and
the controlled Test row. Neither protected attribute may enter model input.

This join creates a separate RQ3-only audit view. `derived_race` remains outside
the controlled Dataset B model view and does not change the 99-column role
arithmetic. The five raw `applicant_race-1` through `applicant_race-5` fields
remain in the controlled audit metadata for traceability, but RQ3 does not use
them to construct race groups or calculate fairness metrics.

The existing shared 1,000-replicate Test bootstrap index artifact is reused.
It must not be regenerated. RQ3 performs no fitting or prediction calls.

## 3. Sex audit

The primary comparison is Female minus Male:

- Male: raw code `1`, reference group;
- Female: raw code `2`, target group;
- codes `3`, `4`, and `6`: retained and reported as coverage-only groups.

Coverage-only rows are not deleted from audit metadata and are not silently
merged into Male, Female, or an `Other` group.

## 4. Derived-race audit

`derived_race` is the HMDA application-level aggregation based on applicant and
co-applicant information. It must not be described as applicant-only race.
Using it keeps the group definition aligned with the documented HMDA
application-level aggregation contract.

White is the reference group. Black or African American and Asian are primary
comparisons. American Indian or Alaska Native, 2 or more minority races, and
Joint are secondary descriptive comparisons subject to the same denominator
rules. Native Hawaiian or Other Pacific Islander, Free Form Text Only, and Race
Not Available are coverage-only under the locked Test counts. Joint must not be
described as an applicant's race.

## 5. Metrics and direction

The positive outcome remains Approved throughout RQ1, RQ2, and RQ3. RQ3 uses
`y_approved`, `p_approved`, and `predicted_approved`; it does not invert labels
or create a Denied-positive production view. For each model and eligible group,
RQ3 reports selection rate, TPR Approved, FPR Approved, FNR Approved, group
Brier score, calibration residual, and absolute calibration residual.
Calibration residual is the group mean of `p_approved - y_approved`: positive
means average approval probability is above the observed approval rate and
negative means it is below. Group Brier score is the group mean of
`(p_approved - y_approved)^2`.

Every comparison uses `target group - reference group`. RQ3 saves both the
signed gap and its absolute magnitude for selection rate, TPR Approved, FPR
Approved, FNR Approved, and calibration residual. The absolute calibration
residual gap is the absolute magnitude of the signed residual gap; it is not
the difference between two group-level absolute residuals. Pairwise Equalized
Odds gap is `max(abs(TPR gap), abs(FPR gap))`. Component rates and signed gaps
must remain available; the Equalized Odds summary is not a replacement for
them.

For `applicant_sex`, the primary pairwise result is Female minus Male. For
`derived_race`, pairwise results compare each primary or secondary target group
with White. RQ3 also reports four global race summaries across White and the
primary and secondary groups that pass the denominator for the component being
summarized:

- `demographic_parity_difference_range`: maximum minus minimum selection rate;
- `tpr_approved_range`: maximum minus minimum TPR Approved;
- `fpr_approved_range`: maximum minus minimum FPR Approved;
- `equalized_odds_difference_range`: the maximum of the TPR and FPR ranges.

Coverage-only race groups remain visible in coverage outputs but never enter a
global range. Pairwise signed gaps answer direction relative to the reference;
global ranges answer the magnitude of the largest eligible multi-group spread.

## 6. Denominators and undefined results

Selection rate, Brier score, and calibration residual require at least 100
group rows. TPR and FNR require at least 30 actual Approved rows. FPR requires
at least 30 actual Denied rows. A gap is valid only when both target and
reference group metrics are valid. EOD is valid only when both TPR and FPR
gaps are valid.

An ineligible result is stored as null with an explicit status and its observed
denominators. It is never replaced by zero and is not used to rank models.

## 7. Uncertainty

The same 1,000 paired Test resamples are used for all models, attributes,
groups, and metrics. Model predictions and thresholds remain fixed. The
denominator rules are applied inside every replicate. A confidence interval is
reported only when at least 950 replicates are valid; otherwise the interval is
undefined and the valid-replicate count remains visible.

Percentile intervals use the 2.5th and 97.5th percentiles with linear
quantiles. No Test resplitting, refitting, reinference, index regeneration, or
replacement replicate is allowed.

## 8. Minimal outputs and acceptance gates

The production directory contains only:

- `audit_metadata.csv`;
- `group_metrics.csv`;
- `gap_metrics.csv`;
- `bootstrap_summary.csv`;
- `runtime.csv`;
- `status.json`.

Acceptance requires: exact ordered Test alignment for all six models; a
complete one-to-one audit join; finite probabilities in `[0, 1]`; frozen
threshold reuse; no fitting, inference, or index generation; passing toy
fixtures for gap direction, denominator handling, undefined metrics,
calibration residual, pairwise gaps, and global race ranges; 1,000 paired
bootstrap rows; and a read-only notebook that loads accepted saved outputs
only.

The Ellis paper may be discussed as a methodological reference, but RQ3 does
not reproduce its Denied-positive orientation or claim an exact replication.

## 9. Accepted production evidence

Canonical implementation ownership is under `src/dataset_b/research/rq3/`;
the former `src/dataset_b/fairness/` paths remain compatibility aliases for
active and frozen RQ4b consumers.

Run `rq3_fairness_20261010_01` is the accepted implementation of this
contract. Its audit view contains all 43,996 ordered Test records. The six
models completed sequentially with `status.json = PASS`, zero fit calls, zero
prediction calls, no threshold change, and no bootstrap-index generation.

The accepted directory contains 588 group-metric rows, 420 pairwise/global
gap rows, 1,008 bootstrap-summary rows, and six runtime rows. Confidence
interval status remains explicit: 708 rows have defined intervals and 300
ineligible or coverage-only rows remain undefined rather than being replaced
by zero. The recorded total audit runtime is 686.7488264 seconds.

The visualization notebook reads these accepted CSV/JSON outputs and the
accepted RQ1 reporting metrics only. Its eight code cells have saved execution
counts `1-8`, ten embedded PNG outputs, no saved error output, and a successful
HTML render. It does not fit a model, run inference, choose a threshold,
recalculate fairness metrics, or execute bootstrap resampling. The production
and validation evidence supports descriptive comparisons of the six frozen
models under this contract; it does not change the causal, legal,
creditworthiness, or default-risk interpretation boundary.
