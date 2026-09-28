# Dataset B — Feature governance contract

## 1. Purpose and ownership

This document is the canonical handbook contract for evidence and decisions
within the Dataset B predictor-candidate universe. It owns:

- the feature-screening sequence and evidence requirements;
- decisions for priority candidates and review candidates;
- the temporary training feature lock;
- raw-versus-derived representation decisions;
- open decision gates and the conditions for a final feature lock;
- the supported business-player mapping for future GSV/Shapley–Lorenz work.

The complete 99-column inventory, cohort, target, exclusions, and audit roles
belong to [Data contract](data_contract.md) and are not repeated here.
Preprocessing implementation, model-family execution, and project progress are
outside this document.

## 2. Candidate and decision layers

The data contract defines 35 predictor candidates in three governance layers:

| Governance layer | Count | Meaning |
| --- | ---: | --- |
| Business-priority candidates | 15 | Screen first using business timing, schema, quality, distribution, and targeted redundancy evidence |
| Evidence-required candidates | 18 | Remain outside the main set unless their additional concept, timing, applicability, stability, and proxy risk are resolved |
| Alternative representations | 2 | Compete with related raw representations; they are not additive features by default |
| **Total** | **35** | **Candidate universe, not a model-input list** |

Four counts must remain distinct:

- **35 candidates:** the complete governance universe;
- **15 priority candidates:** the first screened block;
- **12 `keep` decisions:** the priority fields admitted to the temporary lock;
- **12 raw model inputs:** the machine-readable implementation of that
  temporary lock.

Equality between the last two counts is intentional for the temporary lock. It
must not be generalized into a final feature-selection result.

## 3. Evidence authority and interpretation

Feature claims must be traceable to saved notebook outputs, a versioned config,
or another inspectable artifact. Narrative repetition is not independent
evidence.

The saved evidence is divided across two notebooks:

- `notebooks/eda_datasetB/04_feature_selection.ipynb` contains the executed
  audit and decision outputs for priority groups 4.1–4.6;
- `notebooks/eda_datasetB/05_feature_selction.ipynb` contains the executed
  checkpoint work for review groups 5.1–5.3.

The second notebook does not replace the first. Its initial schema table for
group 5.1 contains an intermediate failed mapping check; later reviewed cells
correct the mapping and publish the gate used by the screening decision. Claims
must use the final reviewed outputs, not the intermediate table.

Neither notebook contains model fitting, prediction, final GSV/SL computation,
`FINAL_MODEL_FEATURES`, or `EXPLANATION_GROUPS`. Saved screening output must not
be described as model-based validation or as a final feature lock.

## 4. Governance sequence

Candidate decisions follow this order:

1. confirm business meaning and prediction-time availability;
2. validate schema, raw states, coverage, and applicable population;
3. inspect predefined redundancy and association blocks;
4. inspect organization/protected-group coverage when proxy risk is relevant;
5. exclude fields that fail timing, leakage, quality, or defensible-scope gates;
6. send unresolved but eligible blocks to an explicitly approved comparison;
7. resolve raw-versus-derived representation choices;
8. version the final raw feature list and business-player mapping;
9. only then permit final downstream deliverables to rely on the final lock.

Association, rarity, missingness, or one model importance score is not sufficient
on its own to keep or exclude a candidate.

### 4.1 Metadata roles and protected-group diagnostics

Feature screening distinguishes two metadata groups; neither group is a
predictor or a GSV/Shapley–Lorenz business player:

| Role | Columns | Permitted use in this contract |
| --- | --- | --- |
| Protected metadata — 6 columns | `applicant_sex`, `applicant_race-1` through `applicant_race-5` | Treat the five race slots as one multi-valued `applicant_race` group; inspect valid codes, slot consistency, group size, coverage, missing/special states, and whether candidate coverage or redundancy conclusions vary by group. |
| Reference/audit metadata — 3 columns | `lei`, `county_code`, `applicant_age` | Preserve for traceability or a separately scoped robustness/audit analysis; do not use these fields to decide whether a predictor is kept in the current screening contract. |

For protected metadata, the minimum descriptive audit records group counts and
shares, special/missing-state rates, candidate-feature coverage by group, and
whether rare-category/missingness patterns or redundancy findings are
concentrated in a group. These diagnostics describe coverage and sensitivity;
they do not establish causality, discrimination, or a fairness conclusion.
All nine columns remain outside model `X`; the complete data-role inventory is
owned by [Data contract](data_contract.md).

### 4.2 Minimum candidate data audit

Before deciding a candidate, record the applicable items below without fitting
the model preprocessing pipeline:

- source dtype, raw tokens, schema-valid domain/codes, and blank, `NA`,
  `Exempt`, `1111`, and not-applicable states separately;
- coverage, parse rate, category count, and rare-category counts;
- availability before the intended decision and any conditional applicable
  population;
- overlap with related candidates or competing representations.

Do not impute, one-hot encode, scale, or fit a model as part of this descriptive
screening audit. Any target/protected-group association must use only the
approved analysis/train scope; the held-out Test partition is not available for
feature decisions.

### 4.3 Predefined redundancy and association blocks

Test only blocks with an explicit overlap hypothesis; do not screen by building
a global correlation matrix and applying one cutoff. The current blocks are:

1. `loan_amount` – `property_value` – `loan_to_value_ratio`;
2. `income` – `debt_to_income_ratio`;
3. `loan_type` + `lien_status` versus `derived_loan_product_type`;
4. `construction_method` + `total_units` versus `derived_dwelling_category`;
5. the three transaction-characteristic flags;
6. the five AUS slots, if AUS remains in scope.

Choose the association diagnostic to match the data types: Spearman correlation
for numeric pairs, Cramér’s V for categorical pairs, correlation ratio or
grouped-distribution comparison for numeric–categorical pairs, and
overlap/co-occurrence or group-level association for multi-valued blocks.
Association measures overlap, not predictive importance or causality, and do
not alone determine keep/exclude.

### 4.4 Sensitivity and grouped-validation gate

Use sensitivity analysis only for decisions that remain ambiguous after
business meaning, timing, data quality, and redundancy review. Candidate
comparisons include `loan_amount + loan_to_value_ratio` versus
`loan_amount + property_value`, presence/absence of the
`transaction_characteristics` block, and each raw-versus-derived pair.
Prefer data-level and association sensitivity. If model-based evidence is
necessary, the described narrow protocol is one Logistic Regression and one
Random Forest comparison on Train/Validation; it is not a six-model run or a
GSV/Shapley–Lorenz run.

Any later grouped validation requires its own approved protocol. Compare
business-block addition/ablation and raw/derived alternatives on matched data
boundaries and hyperparameter budgets; do not add individual slots from one
business concept as if independent. Where required by that protocol, report
variation across seeds/folds, `Δ ROC-AUC`, `Δ PR-AUC` with Denied as the
priority class, a locked calibration measure such as `Δ Brier score`, and
uncertainty for paired differences. Feature importance alone is insufficient.
No model-based sensitivity or grouped-validation result is implied by the
screening evidence summarized in this contract.

## 5. Priority-candidate evidence — 15 fields

The priority notebook records a Dataset B cohort of 293,301 eligible records.
Its saved cells are structurally valid, have contiguous execution counts
`1–50`, and contain no stored runtime error.

All 15 priority candidates have recorded
`timing_status = available_before_decision`. Their saved screening decisions are:

| Decision state | Count | Fields |
| --- | ---: | --- |
| `keep` | 12 | `loan_type`, `loan_purpose`, `lien_status`, `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose`, `income`, `debt_to_income_ratio`, `loan_amount`, `loan_to_value_ratio`, `loan_term`, `occupancy_type` |
| `exclude_from_main` | 1 | `property_value` |
| `keep_for_representation_comparison` | 2 | `construction_method`, `total_units` |

### 5.1 Evidence summary by business block

| Block | Saved evidence | Governance outcome |
| --- | --- | --- |
| Loan structure and purpose | Three fields have valid-code coverage recorded as 100%; each represents a distinct business concept | Keep `loan_type`, `loan_purpose`, and `lien_status`; the product representation comparison remains open |
| Transaction characteristics | Three flags have valid-code coverage recorded as 100%; rare valid codes are preserved and the fields are not deterministic copies of one another | Keep all three raw flags and map them to one shared business player |
| Financial capacity | `income` and `debt_to_income_ratio` retain different scale-versus-burden meanings despite association | Keep both as separate raw fields and separate players |
| Loan and collateral scale | Reported LTV is not an exact arithmetic duplicate of `loan_amount` and `property_value`; retaining all three would preserve overlapping scale information | Keep `loan_amount` and `loan_to_value_ratio`; exclude `property_value` from the main set |
| Loan term | Valid-domain and distribution evidence supports a distinct contract-duration concept | Keep `loan_term` |
| Property characteristics | `occupancy_type` has a completed keep decision; `construction_method` and `total_units` pass quality screening but participate in an unresolved representation choice | Keep `occupancy_type`; hold the other two for comparison |

These are data-quality, distribution, business-timing, and targeted-redundancy
decisions. The notebook does not establish incremental predictive performance or
protected-group sensitivity for the priority block.

### 5.2 Loan structure, purpose, and transaction characteristics

The saved priority-screening output records these distributions:

| Feature | Observed categories and counts | Decision |
| --- | --- | --- |
| `loan_type` | Conventional 265,629 (90.57%); FHA 21,121 (7.20%); VA 6,208 (2.12%); RHS/FSA guaranteed 343 (0.12%) | `keep` |
| `loan_purpose` | Home purchase 139,144 (47.44%); home improvement 49,762 (16.97%); other purpose 40,360 (13.76%); cash-out refinancing 34,595 (11.80%); refinancing 29,412 (10.03%); not applicable 28 (0.01%) | `keep` |
| `lien_status` | First lien 204,431 (69.70%); subordinate lien 88,870 (30.30%) | `keep` |

These fields passed the recorded valid-code checks. Rare valid categories were
retained. The three fields represent distinct concepts; the raw
`loan_type`/`lien_status` versus `derived_loan_product_type` comparison remains
open.

The transaction flags have valid raw states `1`, `2`, and `1111` (`Exempt`).
The saved output records:

| Feature | Informative records | Exempt | Informative values |
| --- | ---: | ---: | --- |
| `reverse_mortgage` | 284,665 (97.06%) | 8,636 (2.94%) | Reverse mortgage 786; not reverse mortgage 283,879 |
| `open-end_line_of_credit` | 284,979 (97.16%) | 8,322 (2.84%) | Open-end 81,148; not open-end 203,831 |
| `business_or_commercial_purpose` | 284,680 (97.06%) | 8,621 (2.94%) | Business/commercial 25,039; not primarily business/commercial 259,641 |

There are 284,656 records (97.05% of the cohort) informative on all three
flags; 8,645 are excluded from this co-occurrence table because at least one
flag is exempt. The six observed patterns, in the order
`reverse_mortgage`–`open-end_line_of_credit`–`business_or_commercial_purpose`,
are:

| Pattern | Records | Share of informative population |
| --- | ---: | ---: |
| `2–2–2` | 179,681 | 63.12% |
| `2–1–2` | 79,152 | 27.81% |
| `2–2–1` | 23,863 | 8.38% |
| `2–1–1` | 1,174 | 0.41% |
| `1–1–2` | 642 | 0.23% |
| `1–2–2` | 144 | 0.05% |

The observed patterns do not show one flag determined by the other two. The
three raw fields remain three model inputs in the temporary lock and map to one
business player, `transaction_characteristics`. `Exempt` remains a separate
category and is not recoded as `2`.

### 5.3 Financial capacity

The saved screening output records 3,007 raw values for `income`: 272,723
numeric (92.98%) and 20,578 `NA` (7.02%). Among numeric values, 360 are
negative, 3,111 are zero, and 269,252 are positive. The reported raw-unit
summary is minimum −11,229; p01 0; Q1 75; median 120; Q3 190; p99 1,242; and
maximum 100,000,000. The long right tail and negative/zero values are
preprocessing considerations; the screening record did not automatically
remove, clip, or log-transform them.

`debt_to_income_ratio` has 21 raw categories. Nineteen informative bands cover
262,193 records (89.39%); 31,108 special values (10.61%) comprise `NA` 22,466
(7.66%) and `Exempt` 8,642 (2.95%). The saved band counts are:

| DTI band | Records | Cohort share | DTI band | Records | Cohort share |
| --- | ---: | ---: | --- | ---: | ---: |
| `<20%` | 15,077 | 5.14% | `20%-<30%` | 36,366 | 12.40% |
| `30%-<36%` | 36,082 | 12.30% | `36` | 6,910 | 2.36% |
| `37` | 7,359 | 2.51% | `38` | 7,751 | 2.64% |
| `39` | 8,295 | 2.83% | `40` | 8,412 | 2.87% |
| `41` | 8,943 | 3.05% | `42` | 9,999 | 3.41% |
| `43` | 8,761 | 2.99% | `44` | 9,959 | 3.40% |
| `45` | 7,729 | 2.64% | `46` | 7,763 | 2.65% |
| `47` | 7,992 | 2.72% | `48` | 8,553 | 2.92% |
| `49` | 12,372 | 4.22% | `50%-60%` | 26,226 | 8.94% |
| `>60%` | 27,644 | 9.43% |  |  |  |

There are 259,767 records (88.57% of the cohort) informative on both income
and DTI. The saved screening summary reports median income decreasing from 196
in the `<20%` band to 75 in the `>60%` band, while income IQR remains positive
in all 19 bands (345 and 72 in those two bands, respectively). The governance
decision keeps both raw fields as separate concepts and players; this is not a
claim that their association was tested for predictive performance.

### 5.4 Loan and collateral scale

The saved output reports these distributions for the three-field overlap
block:

| Feature | Raw unique values | Numeric records | Special values | Minimum | Q1 | Median | Q3 | P99 | Maximum |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `loan_amount` | 993 | 293,301 (100.00%) | 0 | 5,000 | 105,000 | 205,000 | 435,000 | 2,435,000 | 600,855,000 |
| `property_value` | 1,223 | 279,550 (95.31%) | `NA` 5,109 (1.74%); `Exempt` 8,642 (2.95%) | 5,000 | 255,000 | 495,000 | 775,000 | 4,445,000 | 1,400,005,000 |
| `loan_to_value_ratio` | 80,236 | 274,681 (93.65%) | `NA` 9,978 (3.40%); `Exempt` 8,642 (2.95%) | 0.001 | 55.285 | 74.99 | 84.444 | 105.021 | 2,000,000 |

All numeric values in these three fields were positive in the saved screening
output. LTV had 5,261 values above 100 (1.92% of numeric values), 246 above
200 (0.0896%), and 60 above 1,000 (0.0218%); p99.9 was 193.4654. These are
range diagnostics, not automatic invalid-value rules.

| Common numeric population | Records | Cohort share |
| --- | ---: | ---: |
| `loan_amount` and `property_value` | 279,550 | 95.31% |
| `loan_amount` and LTV | 274,681 | 93.65% |
| `property_value` and LTV | 273,800 | 93.35% |
| All three | 273,800 | 93.35% |

On the common population of 273,800, saved Spearman correlations were 0.6080
for `loan_amount`–`property_value`, 0.2377 for `loan_amount`–LTV, and −0.2526
for `property_value`–LTV. Comparing reported LTV with
`loan_amount / property_value × 100` on the same records, 45.25% were within
one point and 66.63% within five points; median absolute difference was
1.285385, p95 was 64.81707, and maximum was 1,999,700. The saved analysis
therefore does not treat reported LTV as an exact arithmetic copy.

The decision keeps `loan_amount` for absolute financing scale and LTV for
relative leverage, and excludes `property_value` from the main set while
retaining it in source/audit. This is a parsimony and interpretability decision;
the screening did not run a model ablation to measure incremental predictive
value. LTV numeric coverage is 1.66 percentage points below `property_value`
(4,869 records); that difference does not imply those records are dropped from
the cohort.

### 5.5 Loan term

The saved output records 338 raw values for `loan_term`, with 274,648 numeric
values (93.64%) and 18,653 special values (6.36%): `NA` 10,360 (3.53%) and
`Exempt` 8,293 (2.83%). Numeric values are positive integers. The reported
raw-unit summary is minimum 1; p01 12; Q1 300; median 360; Q3 360; p99 360;
maximum 900. The most common values are 360: 189,844 (69.12% of numeric
values); 240: 27,838 (10.14%); 300: 20,394 (7.43%); 180: 14,051 (5.12%); and
120: 9,589 (3.49%).

Range diagnostics counted 78 values below 12 (0.0284%), 2,702 above 360
(0.9838%), 9 above 480 (0.0033%), 1 above 600 (0.0004%), and 1,953 not
divisible by 12 (0.7111%). The saved analysis did not infer a time unit,
convert values to months/years, or mark these values invalid. `loan_term`
remains a separate raw feature and business player for contract duration.

### 5.6 Property characteristics

The saved output records the following distributions:

| Feature | Raw categories | Observed distribution | State |
| --- | ---: | --- | --- |
| `occupancy_type` | 3 | Principal residence 257,433 (87.77%); investment property 28,768 (9.81%); second residence 7,100 (2.42%) | `keep` |
| `construction_method` | 2 | Site-built 286,486 (97.68%); manufactured home 6,815 (2.32%) | `keep_for_representation_comparison` |
| `total_units` | 9 | 1 unit 259,784 (88.57%); 2 units 23,905 (8.15%); 3 units 4,534 (1.55%); 5–24 units 2,091 (0.71%); 4 units 1,971 (0.67%); 25–49 units 405 (0.14%); 50–99 units 333 (0.11%); >149 units 170 (0.06%); 100–149 units 108 (0.04%) | `keep_for_representation_comparison` |

The screening preserved `total_units` as schema-defined categorical values; it
did not parse the ranges as continuous numbers. The saved cross-tabulation of
`construction_method` × `total_units` has all 18 combinations, but support is
uneven:

| Construction method | 1 | 2 | 3 | 4 | 5–24 | 25–49 | 50–99 | 100–149 | >149 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Site-built | 253,039 | 23,881 | 4,528 | 1,968 | 2,073 | 398 | 328 | 103 | 168 |
| Manufactured home | 6,745 | 24 | 6 | 3 | 18 | 7 | 5 | 5 | 2 |

The largest cell contains 253,039 records and the smallest contains 2. Having
all combinations present does not establish stable support, so the screening
does not propose an interaction. `occupancy_type` remains `keep`; the choice
between the raw `construction_method`/`total_units` pair and
`derived_dwelling_category` remains open.

### 5.7 Limits of the priority screening

Across the 15 priority fields, the saved output records 100% schema-valid
coverage, while informative coverage is lower for fields with valid special
states. Category imbalance and numeric tails are diagnostics for preprocessing,
not automatic rules to drop rows or fields. The only priority field excluded
from the main set is `property_value`; the two dwelling fields remain in a
representation comparison. These results do not establish target association,
model performance, protected-group sensitivity, causality, or final feature
importance.

## 6. Temporary training feature lock v0.2

`config/dataset_b/feature_set_v1.yaml` publishes the priority block's 12 `keep`
fields as `temporary_training_lock_v0_2` for `rq1_cv`:

1. `loan_type`
2. `loan_purpose`
3. `lien_status`
4. `reverse_mortgage`
5. `open-end_line_of_credit`
6. `business_or_commercial_purpose`
7. `income`
8. `debt_to_income_ratio`
9. `loan_amount`
10. `loan_to_value_ratio`
11. `loan_term`
12. `occupancy_type`

The lock is versioned and machine-readable, but its status remains temporary.
It supports the bounded RQ1/CV work already configured for those inputs; it does
not resolve the remaining 23 candidates and must not be renamed or interpreted
as the final feature set.

`config/features.yaml` delegates Dataset B model inputs to this versioned
contract and likewise labels the Dataset B state as
`temporary_training_lock_v0_2`.

## 7. Current screening state of the 18 review candidates

The review notebook has 24 executed code cells with contiguous execution counts
`1–24` and no stored runtime error. It contains completed checkpoint evidence for
groups 5.1 and 5.2, a deliberately failed timing gate for group 5.3, and no
implementation for groups 5.4 or 5.5.

| Candidate or block | Fields | Evidence state | Governance state |
| --- | --- | --- | --- |
| Conditional preapproval | `preapproval` | Schema, timing, applicability, redundancy, and LEI diagnostics recorded | `advance_to_grouped_validation`; no final decision or player |
| Submission/institution joint block | `submission_of_application`, `initially_payable_to_institution` | Final reviewed schema/timing gate passes; informative-only association recorded | `advance_to_grouped_validation_as_joint_block`; no final decision or player |
| Conforming-loan-limit flag | `conforming_loan_limit` | Public-derived basis differs by action outcome | `exclude_from_further_review`; decision `exclude`; player `not_applicable` |
| Rare special-repayment flags | `negative_amortization`, `other_nonamortizing_features` | Valid schema, but code-1 observations are extremely rare and concentrated in few LEIs | `exclude_from_further_review`; decision `exclude`; player `not_applicable` |
| Interest-only/balloon joint block | `interest_only_payment`, `balloon_payment` | Timing, coverage, informative-only overlap, LEI, and descriptive protected-metadata checks recorded | `advance_to_grouped_validation_as_joint_block`; no final decision or player |
| Manufactured-home/multifamily block | `manufactured_home_secured_property_type`, `manufactured_home_land_property_interest`, `multifamily_affordable_units` | Schema, applicability, redundancy, numeric, LEI, and descriptive protected checks recorded; local prediction-time lineage is not established | `hold_for_timing_lineage_confirmation`; decision and player remain unset |
| Credit-score model types | `applicant_credit_score_type`, `co-applicant_credit_score_type` | No saved screening implementation | Open; no decision or player |
| AUS slots | `aus-1` through `aus-5` | No saved slot-order, schema/timing, or grouped-redundancy implementation | Open; no decision or player |

The 18-field reconciliation is:

| State | Field count |
| --- | ---: |
| Excluded from further review | 3 |
| Advanced to a future approved comparison | 5 |
| Held for timing-lineage confirmation | 3 |
| Not yet screened | 7 |
| **Total** | **18** |

### 7.1 Submission and institutional-relationship block

The saved checkpoint records 100% valid-code coverage for `preapproval`,
`submission_of_application`, `initially_payable_to_institution`, and
`conforming_loan_limit` after treating `NA` for the last field as a valid
not-applicable state. `preapproval` is conditional on home-purchase scope; the
saved output records 9,516 records with requested preapproval, about 6.84% of
home-purchase records.

`submission_of_application` and `initially_payable_to_institution` are advanced
as one joint candidate block. Their saved Cramér's V is 0.7178 on the full
cohort and about 0.1744 on informative-only codes `1` and `2`; the shared
`1111` state materially raises the full-cohort association. The saved LEI
dominance uplift is about 12.60 percentage points for `submission_of_application`
and 4.52 points for `initially_payable_to_institution`; LEI is a diagnostic,
not a predictor.

`conforming_loan_limit` is excluded from further review because its public-
derived flag uses reported loan amount on a basis that differs by action
outcome, so the field is not established as consistently pre-decision safe.
It remains in source/audit; its screening decision is `exclude` and its player
is `not_applicable`.

### 7.2 Special repayment block

All four fields have 100% recorded valid-code coverage; `1111` is a reporting
exemption, not “No.” The saved checkpoint records 8,644 shared `1111` rows and
full-cohort pairwise Cramér's V values around 0.71. On informative-only
records, the highest pairwise value is 0.1701 for
`interest_only_payment` × `balloon_payment`.

The saved output records 401 code-`1` rows for `negative_amortization`, across
four LEIs with 392 rows at one LEI, and 411 code-`1` rows for
`other_nonamortizing_features`, across 18 LEIs. Both are excluded from further
review because of rarity and institutional concentration. The
`interest_only_payment`/`balloon_payment` block advances jointly; its final
keep/exclude decision and business player remain unset.

### 7.3 Manufactured-home and multifamily block

The saved checkpoint reconciles all three fields to 293,301 rows without blank
or invalid tokens. Informative counts are 6,262 for
`manufactured_home_secured_property_type`, 6,258 for
`manufactured_home_land_property_interest`, and 2,434 for
`multifamily_affordable_units`. The context-specific `Not Applicable` counts
are 26, 28, and 163; the corresponding `Exempt` counts are 527, 529, and 510.
These states remain distinct and are not imputed to zero.

For the two manufactured-home fields, saved Cramér's V is 0.9809 for status
within the manufactured-home context and 0.8393 on informative-only values;
they are therefore considered only as a joint business block. All 2,434
informative multifamily values parse as non-negative integers, including 2,091
zeros. The saved LEI row-weighted dominance uplifts over global dominance are
37.3441, 41.1445, and 21.6286 percentage points, respectively. These figures
are proxy-risk diagnostics, not automatic exclusion rules. Protected metadata
checks describe coverage/status only.

HMDA Filing Instructions Guide fields 89, 90, and 92 establish reporting
domains but do not establish local availability at the project's approval
prediction point. Therefore `timing_gate = False`; the three fields remain
`hold_for_timing_lineage_confirmation` with decision and player unset, and
checkpoint 5.3 does not open 5.4.

“Advance” is not a keep decision. No advanced block may enter the main feature
set until its required comparison has been authorized, executed, reviewed, and
recorded.

## 8. Alternative representations — 2 fields

Two raw-versus-derived choices remain open:

| Concept | Raw representation | Derived representation | State |
| --- | --- | --- | --- |
| Loan product | `loan_type` + `lien_status` | `derived_loan_product_type` | Temporary lock uses the raw fields; final representation decision remains open |
| Dwelling | `construction_method` + `total_units` | `derived_dwelling_category` | Both raw fields passed quality screening, but neither raw nor derived representation is final |

Related raw and derived representations must not enter the same main model by
default. A final choice requires comparable data boundaries and an approved,
documented comparison. Preference for interpretability is a decision rule, not
evidence that a comparison has already run.

## 9. Business-player mapping supported by current evidence

The 12 fields in the temporary lock map to 10 documented business players:

| Business player | Raw field or fields |
| --- | --- |
| `loan_type` | `loan_type` |
| `loan_purpose` | `loan_purpose` |
| `lien_status` | `lien_status` |
| `transaction_characteristics` | `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose` |
| `income` | `income` |
| `debt_to_income_ratio` | `debt_to_income_ratio` |
| `loan_amount` | `loan_amount` |
| `loan_to_value_ratio` | `loan_to_value_ratio` |
| `loan_term` | `loan_term` |
| `occupancy_type` | `occupancy_type` |

This mapping is evidence for how the temporary fields can be grouped; it is not
a final `EXPLANATION_GROUPS` artifact and does not establish that GSV/SL has
been executed. Players for unresolved review blocks and dwelling/product
alternatives remain unset.

## 10. Decision rules

### 10.1 Priority candidates

Keep a priority candidate only when it has a defensible pre-decision meaning,
acceptable quality and coverage, and no better selected representation. Exclude
it when timing, leakage, quality, or targeted redundancy evidence provides a
specific reason.

### 10.2 Evidence-required candidates

An evidence-required candidate remains outside the main set unless it:

- represents a distinct and relevant business concept;
- is available at the intended prediction point;
- has interpretable applicability and raw-state semantics;
- does not operate mainly as reporting-process, missingness, or organization
  proxy;
- provides stable value under an approved comparison;
- justifies the additional business-player complexity.

### 10.3 Representation choices

Choose one representation for each raw-versus-derived concept. Prefer the raw
representation when evidence is equivalent and the raw fields are easier to
trace. Choose a derived representation only when the supported benefit is
stable and the concept remains interpretable for downstream explanation.

### 10.4 Audit metadata

Audit metadata may support coverage, organization-proxy, protected-group, and
robustness diagnostics but must not become predictors through a feature-screening
side effect. Use of audit metadata must preserve the role boundary in the Data
Contract.

## 11. Conditions for a final feature lock

A final lock may be created only after all of the following are satisfied:

1. screening is completed for credit-score model types and the AUS group;
2. prediction-time lineage for the manufactured-home/multifamily block is
   resolved, with unresolved fields excluded rather than silently advanced;
3. every field or block advanced from groups 5.1 and 5.2 receives its approved
   comparison and an explicit keep/exclude decision;
4. both raw-versus-derived representation choices are resolved;
5. decision records include business reason, timing, quality, redundancy,
   applicable protected/organization diagnostics, comparison evidence when
   required, decision, reason, and player mapping;
6. the resulting raw feature list and business-player mapping are versioned and
   reviewed together;
7. the final configuration is published under a new identifier rather than by
   relabeling or overwriting `temporary_training_lock_v0_2`;
8. holdout data reserved for final evaluation has not been used to make the
   feature decisions.

Until every condition passes, `FINAL_MODEL_FEATURES` and final
`EXPLANATION_GROUPS` do not exist for Dataset B.

## 12. Required decision-record shape

Every feature or business block considered for the final lock must retain at
least:

| Field | Required meaning |
| --- | --- |
| `feature_or_block` | Raw field or coherent business block |
| `business_reason` | Relevance to the approval prediction problem |
| `timing_status` | Evidence of availability at the prediction point |
| `quality_evidence` | Coverage, valid codes, special states, and applicability |
| `redundancy_evidence` | Related fields or competing representations |
| `organization_proxy_evidence` | LEI/process concentration evidence when relevant |
| `protected_group_sensitivity` | Descriptive coverage/sensitivity evidence when applicable |
| `validation_evidence` | Approved comparison evidence, or an explicit not-run state |
| `stability_evidence` | Variation across the approved resampling scope, when required |
| `decision` | `keep`, `exclude`, or unresolved |
| `decision_reason` | Short evidence-based rationale |
| `gsv_player` | Business player or `not_applicable`; unresolved while the decision is open |

Unset evidence must remain visibly unset. It must not be converted to a passing
value merely to complete the record.

### 12.1 Required logical screening outputs

The completed feature-screening deliverable must provide inspectable logical
outputs for:

- `feature_registry`: candidate fields and their governance roles;
- `priority_feature_audit` and `review_feature_audit`;
- `protected_metadata_audit` for `applicant_sex` and the grouped race slots;
- `redundancy_evidence` for the predefined blocks that were actually screened;
- `feature_decision_table` with evidence, decision, reason, and player mapping;
- `FINAL_MODEL_FEATURES`: the versioned raw columns only after the final-lock
  conditions in §11 pass;
- `EXPLANATION_GROUPS`: the versioned encoded-column-to-business-player map
  only after the same decision process is complete;
- `AUDIT_COLUMNS` and `REFERENCE_METADATA`, retaining their distinct roles.

These are required deliverables, not claims that all outputs already exist.
At the current temporary-lock stage, the saved notebooks do not establish
`FINAL_MODEL_FEATURES` or final `EXPLANATION_GROUPS`.

## 13. Source traceability

Sources used to reconstruct this governance contract:

- [Documentation migration record](../archive/documentation_migration_2026-09.md)
  — identifies `Feature_selection.md` and `Dataset_B_Data_Selection.md` as
  former sources and records their governance, handoff, and data-gate
  destinations;
- `docs/handbook/contracts/data_contract.md` — canonical handbook data roles;
- `notebooks/eda_datasetB/04_feature_selection.ipynb` — executed evidence for
  priority groups 4.1–4.6;
- `notebooks/eda_datasetB/05_feature_selction.ipynb` — executed checkpoint
  evidence for review groups 5.1–5.3;
- `config/dataset_b/feature_set_v1.yaml` — versioned temporary 12-input lock;
- `config/features.yaml` — Dataset B delegation to the versioned feature
  contract.

No standalone feature-screening artifact was found under
`artifacts/dataset_b/`. For this migration, saved notebook outputs and the two
configuration files are the inspectable evidence for the decisions recorded
above.
