# Reproducibility notes

## What can be verified immediately

Run:

```bash
python scripts/verify_public_results.py
```

This checks the compact public CSVs against the manuscript-level headline estimates.

The repository also contains a manuscript evidence freeze:

- `tables/frozen/` — completed manuscript-level analysis tables,
- `tables/derived/` — explicitly labelled secondary/post-hoc descriptive tables,
- `figures/` — six frozen publication figures and their figure manifest,
- `provenance/` — completed-analysis metadata and hash-verification records,
- `EVIDENCE_FREEZE_SUMMARY.md` — interpretation and manuscript wording guardrails.

## Exact scientific source included

The public repository contains the exact scientific source copied from the completed research package:

- `src/` — frozen V1 primary-analysis source,
- `src_expansion/` — secondary-expansion source.

The source was not reconstructed from the manuscript.

## Fixed configurations

The public fixed configuration files are:

- `configs/expanded.json` — expanded analysis, models, seeds and inferential settings,
- `configs/temporal_blocking.json` — UCI HAR overlap-blocking sensitivity settings.

There is no `configs/study_public.json`; older documentation referring to that path was stale and has been corrected.

## Data requirements

The repository does not redistribute third-party source observations. A computational rerun requires obtaining the official datasets from their original repositories:

- UCI HAR: https://doi.org/10.24432/C54S4K
- WISDM: https://doi.org/10.24432/C5HK59

The public source expects the same representations, participant identifiers and preprocessing assumptions documented in `docs/DATASETS.md`, the fixed configs, and source-level metadata.

## Scope of public reproducibility

This repository is designed for:

1. auditing the exact analysis source,
2. inspecting fixed configurations,
3. verifying the manuscript-facing numerical evidence,
4. tracing frozen tables and figures to the completed analysis,
5. reproducing the analysis when the external datasets and required runtime/layout are reconstructed.

It is **not** a byte-for-byte mirror of the internal research workspace and is not presented as a one-command replay of the ~23 GB internal archive.

The public package intentionally omits model checkpoints, the complete raw-prediction archive, interrupted-run logs, local recovery state, and third-party source observations.

## Internal archive relationship

The internal research archive remains the authoritative complete record for fit-level outputs, checkpoints, raw predictions and execution logs. The public repository is the authoritative manuscript-facing snapshot once `v1.0.0` is released.

## Release and DOI

After final repository checks, a `v1.0.0` GitHub release will be created and archived on Zenodo. The Zenodo DOI will then be added to the manuscript and citation metadata.
