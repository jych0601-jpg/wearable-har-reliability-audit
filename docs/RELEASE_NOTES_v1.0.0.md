# v1.0.0 — Manuscript reproducibility release

This release freezes the public manuscript-supporting repository state for:

**Evaluation Protocol Changes Probability Reliability for New-User Wearable Human Activity Recognition**

## Included

- frozen V1 primary-analysis source (`src/`)
- secondary-expansion scientific source (`src_expansion/`)
- fixed analysis configurations
- compact headline results
- manuscript-level frozen tables
- explicitly labelled secondary/post-hoc descriptive tables
- six publication figures and figure manifest
- evidence-freeze provenance and SHA-256 verification records
- reproducibility, dataset, methods and source-scope documentation

## Analysis scope

The primary analysis compares sample-mixed and subject-disjoint evaluation across two public wearable-HAR datasets and four fixed classical probabilistic classifiers.

The secondary robustness expansion includes a fixed 1D-CNN, ten fixed split seeds, temperature and isotonic calibration, selective prediction, adaptive prediction sets, and a UCI HAR overlap-blocking sensitivity analysis.

Post-hoc descriptive manuscript analyses include signed confidence gap, protocol-related model-rank changes, and the UCI primary-seed overlap/residual arithmetic decomposition. These are not retrospectively added to the original primary hypothesis family.

## Public data policy

The release does **not** redistribute UCI HAR or WISDM source observations. Obtain the datasets from their official repositories:

- UCI HAR: https://doi.org/10.24432/C54S4K
- WISDM: https://doi.org/10.24432/C5HK59

The release also excludes the internal ~23 GB research workspace, model checkpoints, the complete raw-prediction archive, interrupted-run logs, and machine-specific recovery state.

## Verification

Run:

```bash
python scripts/verify_public_results.py
```

The manuscript-facing evidence freeze and its provenance records are documented in `EVIDENCE_FREEZE_SUMMARY.md`, `DERIVED_TABLE_METHODS.md`, and `provenance/`.

## DOI

This GitHub release is intended for archival on Zenodo. The resulting Zenodo DOI will be added to the manuscript and repository citation metadata after minting.
