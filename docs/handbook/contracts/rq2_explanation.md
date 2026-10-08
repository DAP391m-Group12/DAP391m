# Dataset B RQ2 explanation contract

## 1. Purpose and status

This contract defines the locked Dataset B RQ2 explanation protocol for
agreement and stability between Global Shapley Value (GSV) and
Shapley--Lorenz (SL). It is implementation guidance only. No RQ2 execution,
result, ranking, runtime, confidence interval, or model-selection conclusion
is established by this document.

The machine-readable source for the same protocol is
[`config/dataset_b/rq2_protocol.yaml`](../../../config/dataset_b/rq2_protocol.yaml).
If this contract and that YAML disagree, the discrepancy is a blocking
documentation defect; do not run RQ2 until it is resolved.

RQ2 reads frozen final artifacts only. It must not fit or mutate a model,
preprocessor, threshold, split, or RQ1 artifact.

## 2. Scope

RQ2 covers these six frozen final runs:

| Model key | Final run ID |
| --- | --- |
| `lr` | `lr_final_15f_20261002_01` |
| `rf` | `rf_final_15f_20261002_01` |
| `xgboost` | `xgb_final_15f_20261003_01` |
| `plain_mlp` | `plain_mlp_final_15f_20261004_03` |
| `realmlp` | `realmlp_final_15f_20261004_01` |
| `tabmmini` | `tabmmini_final_15f_20261004_01` |

The explained output is `p_approved = P(y_approved = 1)`. RQ2 does not
explain thresholded class labels, narrow the scope to a neural-model subset,
or use a superseded 12-feature artifact.

The players are the 12 business players in [Feature governance](feature_governance.md#9-business-player-mapping-supported-by-current-evidence).
They are raw-information coalitions, not encoded-column players. If a player
contains multiple raw fields, all of its member fields enter or leave a
coalition together.

## 3. Samples and reproducibility

The protocol uses a stratified sample of 256 Validation rows as explanation
observations and a stratified pool of 128 Train rows as background donors.
`root_seed = 42` and the derived seeds `4201`, `4202`, and `4203` govern,
respectively, explanation-row selection, background-pool selection, and the
fixed donor assignment.

Each explanation observation is paired once with a background donor. For a
coalition, present players retain their values from the explanation observation
and absent players use the paired donor values. This is interventional
background replacement. It describes frozen-model behavior under this defined
replacement rule; it is not a causal effect.

RQ2 uses Validation for explanation observations. It must not use Test rows
for explanation or rerun final Test inference.

## 4. Exact coalition prediction engine

With 12 players, the engine enumerates every subset: `2^12 = 4,096`
coalitions. For each frozen model and coalition, it constructs raw-player
inputs, applies the frozen model-specific preprocessing, and obtains
`p_approved` through frozen inference only.

The canonical cached object is a coalition prediction cube:

```text
P[coalition, explanation_observation] = p_approved
```

The engine is exact with respect to the 12-player subset enumeration and the
fixed sampled observation/background design. It does not claim integration over
the full data distribution.

Before calculations are accepted, every generated input must be accepted by
the frozen preprocessing route and every prediction must be finite and within
`[0, 1]`. A failure stops the RQ2 run. The implementation must not repair,
coerce, resample, or substitute a coalition input or prediction.

## 5. Global Shapley Value

For model `m`, explanation observation `i`, player `j`, and coalition `S`
that excludes `j`, local GSV is the exact Shapley-weighted difference in
`p_approved` when `j` is added to `S`, calculated from the cached cube.

Global importance is:

```text
I_m,j = mean_i(abs(phi_m,i,j))
W_m,j = I_m,j / sum_j(I_m,j)
```

`W_m,j` determines the GSV importance ranking. The directional summary,
reported separately from importance, is `mean_i(phi_m,i,j)`. This distinction
prevents positive and negative local effects from cancelling in the importance
rank.

## 6. Shapley--Lorenz

SL is a global Shapley decomposition whose coalition value is the
Lorenz-Zonoid score of the coalition prediction vector over the explanation
observations. For each player, SL applies the exact Shapley weights to the
coalition-score differences across all 4,096 subsets.

The implementation must use the original `LZ_{d=1}` definition and
Shapley--Lorenz decomposition of Giudici and Raffinetti (2021), with no
unrecorded surrogate. Before Dataset B execution, it must pass a documented
toy-fixture test that verifies the Lorenz-Zonoid function and the exact
decomposition. This is an implementation acceptance gate, not evidence that
RQ2 has been executed.

The SL bootstrap recomputes its global coalition scores and exact decomposition
from the cached prediction cube. It does not retrain or reinfer a frozen model.

## 7. Ranking, agreement, and uncertainty

For both methods, normalized importance is ranked in descending order. Ties
use average ranks; a Top-3 set includes every player tied at the third-place
cutoff.

Within each model, compare GSV and SL ranks with:

- Spearman correlation;
- Kendall tau-b;
- Top-3 Jaccard similarity; and
- mean absolute rank difference.

For each method separately, cross-model stability reports pairwise Spearman
and Kendall tau-b across the six model rankings. This is descriptive agreement,
not a claim that one model is superior.

Use 1,000 shared bootstrap resamples of explanation-observation indices and
95% percentile confidence intervals. GSV bootstraps the cached local
contribution matrix. SL bootstraps the explanation-observation axis of the
cached prediction cube and recomputes its global statistic. Bootstrap must not
call model fitting or model inference.

## 8. Required artifacts and acceptance gates

Each model/method publication package must retain:

- a copy of the protocol;
- explanation-sample, background-pool, and donor-assignment manifests;
- the coalition prediction cube;
- GSV local contributions;
- global importances, direction summaries, and rankings;
- bootstrap agreement metrics and confidence intervals;
- a runtime log; and
- a validation report.

Acceptance requires: final run identity and frozen-marker verification;
ordered `record_id` manifests; player mapping equality with feature governance;
sample/background sizes and seeds matching the YAML; exactly 4,096 coalitions;
finite range-valid probabilities; no fit or model mutation; and the SL toy
fixture passing. Failure of any gate means RQ2 is not accepted.

## 9. Interpretation boundaries

RQ2 is an explanatory analysis of historical lender approval predictions. It
does not measure default risk, creditworthiness, legal discrimination, causal
effects, or fairness. Agreement or stability between explanation methods does
not establish that either method reveals a true causal feature effect.

## 10. References

- Giudici, P., & Raffinetti, E. (2021). *Shapley-Lorenz eXplainable Artificial Intelligence*. Expert Systems with Applications, 167, 114104. https://doi.org/10.1016/j.eswa.2020.114104
- Agarwal, S., Muckley, C. B., & Neelakantan, P. (2023). *Countering racial discrimination in algorithmic lending: A case for model-agnostic interpretation methods*. Economics Letters, 226, 111117. https://doi.org/10.1016/j.econlet.2023.111117
