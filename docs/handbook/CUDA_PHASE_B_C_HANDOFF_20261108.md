# Dataset B CUDA Phase B and Phase C Handoff

## Purpose

This handoff transfers the current Dataset B reproduction state to the qualified
CUDA host. Continue the stopped Phase B verification, then build and publish the
release bundles in Phase C. This handoff does not authorize training, fitting,
regeneration of frozen experimental artifacts, or changes to locked numerical
policies.

## Current state

- Local source started from commit `4bdfa0a5aebe138982b75a055891c1321a4d5496`.
- GitHub `main` had advanced to `181d5af970e01934fd83c4d2cf33ea2f808fb99e`
  when this handoff was prepared. Use the handoff branch named below; do not reset
  or force-push `main`.
- The reconciled core manifest has 654 assets. The latest local asset verification
  passed 654/654 with no missing or mismatched assets.
- The 2026-11-05 evaluation attempt stopped before any model worker or inference
  process because the current host lacked the five locked environments and CUDA.
  This was not a checkpoint or metric failure.
- `compare`, RQ2, `reports`, the clean-clone gate, public download verification,
  and Phase C were not completed in that attempt.
- The core RQ3 bundle exists and matches the reconciled manifest:
  `dataset_b_rq3_part01.zip`, 266,084 bytes,
  SHA-256 `b2a553143fbed46a822fe6b6b0efee8e0a5cdfa94aa1bd49578e26a140ebef36`.
- The optional RQ4 audit archive also exists and matches its separate manifest:
  `dataset_b_rq4_audit_evidence_20261105.zip`, 1,830,814 bytes,
  SHA-256 `f8bb82202ec167c51e16d3299b808d0af014d76dce67d2c38ac20200dbb6a40b`.
- The other ten core ZIP bundles are not present in this workspace. Every source
  member named by the current manifest was present when checked on 2026-11-08.
- A 2026-11-08 attempt at commit
  `5a5667b13546e6b088756cd8e9da8e963a0ccfe0` passed 654/654 assets, all five
  locked environment fingerprints, all three finite CUDA tensor checks, six of six
  frozen-checkpoint evaluations, and 402/402 RQ1 numerical comparisons. RQ3, RQ4a,
  and RQ4b passed their reproduction comparisons. An initial RQ2 run completed all
  six frozen CPU-inference models but failed 3/28 comparison rows, all for TabMmini:
  `importance.csv` delta `2.12172445194711e-10`, `cube.npy` maximum delta
  `5.960464477539063e-08`, and `gsv_local.npy` maximum delta
  `1.9868215961338365e-09`, each against the locked `1e-12` tolerance. The policy
  and accepted artifacts were not changed. Diagnosis isolated the current-host cause
  to the unpinned Torch CPU intra-op thread count. After a scoped one-thread inference
  lock, `research --route all` passed RQ2 28/28, RQ3 5/5, RQ4a 4/4, and RQ4b 17/17.
  `reports` then returned the allowed documented `INCOMPLETE` result with 5 objects
  PASS and 3 `NOT_REGENERATED`. Phase B is complete; Phase C, LFS staging, commit,
  and push did not run.
- On 2026-11-09 Phase C rebuilt all 11 core ZIPs, totaling 678,068,350 bytes.
  Bundle size, SHA-256, and exact member inventory all matched the prebuild manifest.
  RQ3 `runtime.csv` and the optional RQ4 audit archive remained outside core. Public
  URLs, anonymous download, and clean-clone installation remain unverified. Release
  commit `1d384482f7b00f84c06cdecbb4abd13380da91a2` and all core LFS objects were
  pushed to `codex/cuda-phase-b-c-handoff`; post-push `git lfs fsck` passed.
- The final default data route treats raw HMDA as optional provenance and the three
  locked split CSVs as required direct downloads with atomic `.part` publication and
  exact size/SHA-256 verification. Their URL fields remain null because no usable URLs
  were supplied; the route therefore blocks rather than rebuilding from live raw data.

## Branch and transfer

The handoff branch is `codex/cuda-phase-b-c-handoff`.

On the CUDA host:

```powershell
git fetch origin
git switch --track origin/codex/cuda-phase-b-c-handoff
git lfs install
git lfs pull
git status --short --branch
```

Git carries source, documentation, receipts, and release ZIPs tracked with LFS.
The ignored `data/`, `artifacts/`, environments, `scratch/`, and `refactor/` trees
are not transferred by `git pull`. Before Phase B, the CUDA host must already have
the retained assets at the repository-relative paths in the core manifest and the
five locked environments, or must obtain them through an independently verified
handoff. Do not copy or inspect local `scratch/` as part of the core route.

## Locked boundaries

- Do not train or fit a model.
- Do not overwrite raw data, frozen run roots, checkpoints, predictions,
  thresholds, split membership, or locked bootstrap indices.
- Do not use Test for selection.
- Do not fall back from a CUDA profile to CPU.
- Stop at the first failed gate and retain the generated receipt.
- Treat RQ3 `runtime.csv` as excluded volatile historical timing metadata. The
  reconciled core RQ3 bundle contains five deterministic members.
- Keep `scratch/` and `refactor/` local and out of commits.

## Phase B sequence

Run from the repository root. Use `--env-root` consistently if the five profiles
are not under `.venv-repro/`.

```powershell
python -m src.dataset_b.reproduce verify-assets
python -m src.dataset_b.reproduce environment
python -m src.dataset_b.reproduce evaluate --model all
python -m src.dataset_b.reproduce compare
python -m src.dataset_b.reproduce research --route all
python -m src.dataset_b.reproduce reports
```

Required interpretation:

1. `verify-assets` must pass before any model process starts.
2. `environment` must pass all five profiles, including finite CUDA tensor checks
   for the three neural profiles.
3. `evaluate` must use frozen checkpoints only and must report no fit.
4. `compare` must compare the newly generated outputs with accepted references.
5. `research --route all` includes the previously unrun RQ2 route. Retain its
   receipt and do not interpret an RQ comparison as new training evidence.
6. `reports` may remain `INCOMPLETE` for documented non-regenerated objects. Record
   that result exactly; do not relabel it as a model or metric failure.

Do not continue to Phase C after a failed asset, environment, evaluation,
comparison, or research gate. A documented `reports` limitation may be carried
forward only if all preceding scientific gates pass and the limitation remains the
same object-level regeneration gap already declared in `REPRODUCE.md`.

## Phase C bundle build

Phase C does not require CUDA, but it requires the complete retained source
inventory. The maintainer command deterministically rebuilds all eleven core ZIPs
and rewrites the unpublished manifest from the retained evidence:

```powershell
python -m src.dataset_b.reproduce maintainer-build --source-root .
```

The command rebuilds RQ3 as well as the ten missing bundles. After it completes,
confirm that the bundle count is 11 and verify every ZIP against the newly written
manifest before staging anything:

```powershell
$manifest = Get-Content -Raw config/dataset_b/reproducibility/asset_manifest.json | ConvertFrom-Json
foreach ($bundle in $manifest.bundles) {
    $path = Join-Path dist/reproducibility $bundle.filename
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    [pscustomobject]@{
        File = $bundle.filename
        SizePass = $item.Length -eq $bundle.byte_size
        HashPass = $hash -eq $bundle.sha256
    }
}
```

All 11 rows must have `SizePass=True` and `HashPass=True`. If a bundle differs,
stop and retain both the old manifest and observed identity. Do not silently update
an accepted identity merely to make the check pass.

## Publishing the remaining bundles

The release ZIP paths are configured for Git LFS. After Phase B and Phase C pass:

```powershell
git status --short
git lfs status
git add config/dataset_b/reproducibility/asset_manifest.json
git add dist/reproducibility/*.zip
git add reports/reproducibility/receipts
git commit -m "chore(dataset-b): complete CUDA reproduction handoff"
git push origin HEAD:codex/cuda-phase-b-c-handoff
```

Do not add `data/`, `artifacts/`, `.venv-repro/`, `scratch/`, or `refactor/`.
The eleven core bundles total about 678 MB and the RF bundle is about 348 MB, so a
successful push depends on GitHub LFS authorization and quota. If LFS rejects the
upload, keep the commit and ZIPs unchanged, upload the exact ZIP bytes to the chosen
release store, fill `bundle_links.json`, bind the links, and run
`maintainer-check-publication`. Do not commit credentials or private tokens.

## Evidence to return

Return the following to the original host or commit it on the handoff branch:

- all new receipts under `reports/reproducibility/receipts/`;
- Phase B command outputs and exact exit codes;
- the final 11-bundle filename, byte-size, and SHA-256 table;
- the updated asset manifest and source revision;
- the Git commit and branch containing the LFS pointers;
- any blocked or incomplete gate without changing its status wording.

Completion of Phase B/C must remain separate from clean-clone and anonymous public
download claims until those routes are actually executed.
