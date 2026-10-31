# Configuration Guide

`config/project.yaml` is the only default project-level configuration. It
identifies Dataset B as the active dataset and routes readers to the Dataset B
catalog without duplicating model or research-method parameters.

## Entry points

| Path | Role | Change policy |
|---|---|---|
| `project.yaml` | Project identity, root paths, logging, catalog pointer | Editable only through a scoped config task |
| `dataset_b/catalog.yaml` | Navigation registry for Dataset B contracts | Paths only; do not copy contract values into it |
| `dataset_b/README.md` | Ownership, status, and modification guide | Keep aligned with the catalog and actual files |

Load the project configuration with `load_config()` and a specific Dataset B
contract with `load_config("dataset_b/<name>.yaml")`. Paths stored in these
files are repository-root relative unless the owning contract says otherwise.

Configuration files are not runtime history. Do not append training results,
timestamps, fitted parameters, checkpoint state, or accepted metrics to this
directory. Those belong to their authorized artifact/run evidence.

Do not move or edit a lineage-sensitive Dataset B YAML merely to improve
folder layout. Physical grouping under `contracts/`, `models/`, `research/`,
or `runtime/` is deferred until its reference and frozen-lineage gates pass.
