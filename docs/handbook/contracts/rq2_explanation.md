# Dataset B RQ2 explanation contract

## 1. Purpose and current status

RQ2 compares Global Shapley Value (GSV) with the interventional
Shapley--Lorenz (SL) adaptation for the six frozen final Dataset B models.
The implementation is complete and all six frozen adapters pass the CPU smoke
gate against stored Validation predictions. The full XGBoost CPU benchmark
passes and locks coalition batch size `16`. Six-model production run
`rq2_cpu_production_20261009_01` and its independent artifact validation pass,
so RQ2 is `COMPLETE`. Production bootstrap uses resumable
100-replicate checkpoints and ten ordered chunks per model. GPU/CUDA remains part of the
historical training and freeze evidence for the neural models, but it is not an
RQ2 execution requirement. RQ2 loads the frozen learned state into CPU
inference objects in memory without rewriting the bundles. This does not change
the GSV or SL method.

The machine-readable parameters are in
[`config/dataset_b/rq2_protocol.yaml`](../../../config/dataset_b/rq2_protocol.yaml).
The authoritative current status is registered in [`STATUS.md`](../STATUS.md);
Section 9 below records the synchronized production-closure evidence.

RQ2 may load frozen bundles and run inference. It must not fit a model, select
or change a threshold, mutate a frozen bundle, use Test rows, or rerun model
training.

## 2. Frozen models and explained output

| Model key | Frozen run ID |
| --- | --- |
| `lr` | `lr_final_15f_20261002_01` |
| `rf` | `rf_final_15f_20261002_01` |
| `xgboost` | `xgb_final_15f_20261003_01` |
| `plain_mlp` | `plain_mlp_final_15f_20261004_03` |
| `realmlp` | `realmlp_final_15f_20261004_01` |
| `tabmmini` | `tabmmini_final_15f_20261004_01` |

The explained output is unthresholded `p_approved = P(y_approved = 1)`.
Thresholded class labels are not an explanation target.

## 3. Players, samples, and donors

The game has 12 ordered business players. Together they partition the ordered
15 raw model-input fields. When a player contains several raw fields, all of
those fields enter or leave a coalition together.

The ordered players are `loan_type`, `loan_purpose`, `lien_status`,
`transaction_characteristics`, `income`, `debt_to_income_ratio`,
`loan_amount`, `loan_to_value_ratio`, `loan_term`, `occupancy_type`,
`preapproval`, and `special_repayment_structure`. Bit `j` of a coalition mask
corresponds to player `j` in this order.

The explanation sample contains 256 stratified Validation rows selected with
seed `4201`. The background pool contains 128 stratified Train rows selected
with seed `4202`. Sampling uses `numpy.random.Generator(PCG64(seed))`,
proportional largest-remainder allocation across target strata `[0, 1]`,
sampling without replacement within a stratum, and ascending source position
after selection.

Each explanation observation receives one fixed donor. Seed `4203` draws one
background-pool index per observation with replacement. Present players retain
the explanation-observation values; absent players simultaneously take all of
their raw fields from that observation's donor. Bootstrap resampling preserves
this pairing and never draws a new donor.

`sample.csv` records the selected Validation IDs, target, seed, positions, and
15 input fields. `donors.csv` records the explanation ID, donor ID, donor-pool
index, seeds, pool size, and donor input fields directly.

## 4. Input and predictor gates

Before inference, the runner requires:

1. Nonblank, unique `record_id` values.
2. `y_approved` containing only `0/1` and both strata.
3. The ordered final 15 input fields.
4. Explanation rows from Validation and background rows from Train.
5. Exactly 256 explanation rows and 128 background rows in production.
6. Persisted sample/donor IDs and seeds.
7. One finite `float64` probability per predictor input row, within `[0, 1]`.
8. A smoke comparison against frozen stored Validation predictions.

All six RQ2 adapters use CPU inference. CUDA fallback is prohibited. For
PlainMLP and RealMLP, CPU adaptation changes only the device of the newly
loaded in-memory inference object; learned parameters and frozen files remain
unchanged.

The smoke gate joins by `record_id`, runs the same frozen preprocessing and
class-probability route used by the cube, and requires fresh `p_approved` to
match frozen stored Validation predictions within the configured tolerance.
This gate is necessary because valid raw input alone does not prove that an
adapter reads the approved class or preprocessing route correctly.

## 5. Exact coalition cube and GSV

All `2^12 = 4,096` coalitions are enumerated in integer mask order. For every
model, the cached cube has shape `[4096, 256]`, dtype `float64`, only finite
values, and values within `[0, 1]`.

The locked production coalition batch size is `16`. Benchmark run
`rq2_xgboost_cpu_batch16_20261009T052726780650Z` completed the full XGBoost
cube and 1,000-replicate cache-only bootstrap on CPU. Active compute time was
`1081.29` seconds, including `1051.26` seconds for bootstrap, and peak working
set was `674,009,088` bytes. All four agreement metrics had `1000/1000` valid
replicates. The bootstrap was checkpointed as ten sequential 100-replicate
cache-only calls and concatenated in original replicate order after an earlier
long single-process attempt ended without a final report. This does not change
the bootstrap matrix or statistics, but the current runner's one-call
1,000-replicate process lifecycle is not yet demonstrated on this execution
host. The benchmark establishes numerical/runtime feasibility and the
batch-size lock; it is not a six-model RQ2 result and is not imported as a
production model result. No additional model-specific benchmark is required.

For each observation and player, exact GSV uses the standard Shapley-weighted
average of coalition prediction differences. Its efficiency identity is:

```text
sum(player local contributions)
= full-coalition prediction - empty-coalition prediction
```

Global GSV importance is the mean absolute local contribution across the 256
observations. Mean signed local contribution is retained separately. Importance
is normalized only after the global absolute values are calculated.

## 6. Interventional Shapley--Lorenz

SL is calculated from the same cached prediction cube; it does not call a
model again. For each coalition prediction vector, the one-dimensional
Lorenz--Zonoid value uses average ranks, arithmetic mean, and population
covariance. Exact Shapley decomposition is then applied over all coalitions.

The signed contribution describes the direction of the Lorenz--Zonoid score,
not the direction of `p_approved`. SL importance is its absolute value. The
exact efficiency identity requires the player contributions to sum to the
full-coalition value minus the empty-coalition value.

## 7. Ranking, agreement, and shared bootstrap

Each method is ranked separately in descending normalized importance. Ties use
average ranks. Top-3 includes every player tied at the cutoff.

Within each model, GSV and SL are compared using Spearman correlation, Kendall
tau-b, Top-3 Jaccard, and mean absolute rank difference. Cross-model stability
uses pairwise Spearman and Kendall tau-b separately for GSV and SL.

One matrix of shape `[1000, 256]`, generated with seed `4204`, is shared by
both methods and all six models. GSV resamples the observation axis of cached
local contributions. SL resamples the observation axis of the cached cube and
recomputes only the SL statistic. Bootstrap must not fit or call inference.

The runner reports 95% linear percentile intervals, valid/invalid replicate
counts, and invalid-reason counts. At least 950 replicates must be valid for a
reported interval. An invalid replicate is not replaced.

Production bootstrap partitions the locked matrix into ten consecutive chunks:
`0–99`, `100–199`, ..., `900–999`. Every chunk calls the unchanged cached
bootstrap implementation with the corresponding rows of the shared index
matrix. It writes four `float64` metric distributions and identity/count
metadata using a temporary sibling followed by an atomic file replacement.
This safe per-file checkpoint write is not whole-package atomic publication.

On resume, the runner directly revalidates the deterministic sample and donor
tables, shared bootstrap matrix, analysis/model/frozen-run identity, protocol
version, cached cube and GSV values, chunk bounds, metric keys, shapes, finite
or NaN payload, and valid/invalid reason counts. A valid chunk is skipped. A
missing chunk is calculated from the cached cube and GSV values without loading
a predictor. A partial, corrupt, or mismatched checkpoint fails closed and is
not silently overwritten. Percentile intervals are calculated only after all
ten distributions are concatenated in original replicate order.

Each model retains `status.json`. `PASS` is written only after its cube, exact
methods, all ten checkpoints, merged bootstrap output, and numerical gates
pass. A failure records the model and stage, stops the run, and preserves valid
checkpoints. A resumed `PASS` model is validated and does not load its frozen
predictor. Terminal progress is flushed at run, model, smoke, cube, exact
method, every bootstrap chunk, model completion, failure, and final-run events.

## 8. Minimal output contract

Every execution uses a new `run_id`. An existing output directory is rejected
before any model is loaded. The only required outputs are:

```text
artifacts/dataset_b/rq2/<run_id>/
├── sample.csv
├── donors.csv
├── bootstrap_indices.npy
├── runtime.csv
├── cross_model_stability.csv
└── models/
    ├── lr/
    │   ├── cube.npy
    │   ├── gsv_local.npy
    │   ├── importance.csv
    │   ├── bootstrap.csv
    │   ├── status.json
    │   └── bootstrap_checkpoints/
    │       ├── chunk_000.{npz,json}
    │       └── ... chunk_009.{npz,json}
    └── ... five remaining models
```

`importance.csv` contains both methods, full-precision importance, normalized
importance, average-tie rank, signed summary, player identity, and frozen model
run ID. `bootstrap.csv` contains the four agreement point estimates, confidence
intervals, availability flags, and valid/invalid counts. `runtime.csv` retains
only cube, method, bootstrap, and total seconds per model. Checkpoints remain
part of the production evidence so an interrupted run can resume without
recomputing valid chunks.

No protocol copy, duplicate schema package, standalone validation package, or
content-digest package is required.

## 9. Verified production evidence

The accepted production output is
`artifacts/dataset_b/rq2/rq2_cpu_production_20261009_01/`. Final artifact
validation passed 91 checks with no failures. It confirmed the deterministic
256-row Validation sample, 128-row Train donor pool, shared bootstrap index
shape `[1000, 256]`, and no Test IDs. For every model, it confirmed a finite
range-valid `float64` cube of shape `[4096, 256]`, the GSV local matrix, 24
importance rows, four bootstrap metric rows with `1000/1000` valid replicates,
ten complete checkpoint pairs, and `status.json = PASS`. The cross-model table
contains exactly 60 rows. Benchmark artifacts were not imported into the
production directory, and the six frozen model artifacts were not rewritten.

| Model | Cube seconds | Method seconds | Bootstrap seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| `lr` | 12.0827241 | 3.6924516 | 767.1701952 | 782.9453709 |
| `rf` | 46.2054014 | 3.4441762 | 815.5074962 | 865.1570738 |
| `xgboost` | 19.6574835 | 6.0036580 | 722.7583381 | 748.4194796 |
| `plain_mlp` | 20.8313495 | 3.9898250 | 688.9451561 | 713.7663306 |
| `realmlp` | 46.6543562 | 4.0514448 | 858.2302958 | 908.9360968 |
| `tabmmini` | 115.9317464 | 4.2814589 | 677.9804279 | 798.1936332 |
| **Total** | **261.3630611** | **25.4630145** | **4530.5919093** | **4817.4179849** |

The verified results notebook is
[`notebooks/dataset_b/08_rq2_gsv_sl_results.ipynb`](../../../notebooks/dataset_b/08_rq2_gsv_sl_results.ipynb).
Its saved execution reads the accepted production CSV/JSON artifacts only; it
does not load a model, run inference, construct coalitions, or recompute GSV,
SL, or bootstrap distributions.

## 10. Notebook boundary

`notebooks/dataset_b/08_rq2_gsv_sl_results.ipynb` may read the saved outputs,
build importance/ranking tables, plot GSV versus SL, plot agreement intervals,
compare six-model stability, show runtime, and warn about invalid bootstrap
replicates. It must not load a model, build coalitions, run inference, or
recompute bootstrap results.

## 11. Interpretation boundary

RQ2 describes frozen-model behavior for historical mortgage-application
approval under the documented replacement game. It does not measure default
risk, creditworthiness, legal discrimination, causal effects, or fairness.
Agreement or stability does not establish explanation correctness or ground
truth.
