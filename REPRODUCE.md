# Reproduction and Verification Router

This repository preserves completed Dataset B experiments and their accepted
evidence. Start with the route that matches the work being performed; do not
rerun a completed experiment merely to verify the handover refactor.

## Read-only handover verification

The public Dataset B interface provides CPU-safe navigation checks:

```text
python -m src.dataset_b.cli status
python -m src.dataset_b.cli validate-config
python -m src.dataset_b.cli validate-artifacts --scope frozen
python -m src.dataset_b.cli validate-notebooks --scope final
python -m src.dataset_b.cli report-index
```

These commands inspect repository structure and frozen markers. They do not
fit or load models, run inference, select thresholds, regenerate bootstrap
results, execute notebooks, or publish artifacts.

## Dataset B method and execution routes

- [`docs/handbook/README.md`](docs/handbook/README.md) routes to the applicable
  data, feature, preprocessing, training, evaluation, and research contracts.
- [`docs/handbook/STATUS.md`](docs/handbook/STATUS.md) is the sole authority for
  completed work, accepted runs, open gates, and runtime evidence.
- [`config/README.md`](config/README.md) explains project and Dataset B
  configuration discovery.
- [`notebooks/dataset_b/README.md`](notebooks/dataset_b/README.md) distinguishes
  historical notebooks from current reporting notebooks.
- [`src/dataset_b/README.md`](src/dataset_b/README.md) documents canonical
  source ownership and compatibility paths.

Before any authorized model-family execution, read the applicable handbook
contract and runbook. Preserve the immutable split and raw-data boundaries,
fit preprocessing on Train only, select thresholds on Validation only, and
reserve Test for the documented final evaluation. Existing frozen artifacts,
checkpoints, predictions, thresholds, bootstrap indices, and accepted results
must not be overwritten.

GPU/CUDA absence is not a failure of the repository handover. GPU-dependent
routes require their documented environment and explicit task authorization;
the handover gate uses code review, structure/import checks, and CPU-safe or
mocked tests without weakening production GPU guards.
