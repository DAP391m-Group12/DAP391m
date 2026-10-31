# Dataset B Configuration Index

This directory contains the Dataset B configuration contracts retained at
their established paths for handover and frozen-lineage compatibility.
`catalog.yaml` is a navigation registry, not a second source of method values.

## Files

| File | Logical owner | Status | Modification policy |
|---|---|---|---|
| `feature_set_v1.yaml` | contracts/features | active, lineage-sensitive | Do not edit without a feature-contract and lineage task |
| `split_v1.yaml` | contracts/split | active, widely referenced | Keep path and semantics stable |
| `model_protocol_v1.yaml` | models/classical | active contract | Audit consumers before any move or edit |
| `plain_mlp_v1.yaml` | models/plain_mlp | final-lineage contract | Preserve until lineage audit authorizes change |
| `realmlp_v1.yaml` | models/realmlp | final-lineage contract | Preserve; RQ4b references this path |
| `tabmmini_v1.yaml` | models/tabmmini | final-lineage contract | Preserve until lineage audit authorizes change |
| `rq2_protocol.yaml` | research/rq2 | production authority | Preserve path and method values |
| `rq3_protocol.yaml` | research/rq3 | production authority | Preserve; RQ4 also consumes its definitions |
| `rq4a_postprocessing.yaml` | research/rq4a | production contract | Preserve until manifest/hash audit |
| `rq4b_adversarial.yaml` | research/rq4b | frozen Test-manifest hash | Do not move or edit |
| `runtime_colab_gpu_v1.yaml` | runtime | historical runtime profile | Not a scientific method authority |
| `catalog.yaml` | navigation | active registry | Paths only; every target must exist |

## Consumer rules

- Read project defaults from `config/project.yaml`.
- Use `catalog.yaml` to discover the authoritative file for a logical role.
- Read scientific values from the target contract, never from this README or
  duplicated catalog fields.
- Do not create duplicate YAML copies or symlink aliases.
- Do not write training history or runtime state into configuration files.
- Check artifact manifests, receipts, source pins, documentation, notebooks,
  and tests before changing a lineage-sensitive path.
