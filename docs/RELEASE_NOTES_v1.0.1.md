# v1.0.1 — Manuscript reproducibility release

This release supersedes v1.0.0 for archival and Zenodo citation purposes.

The scientific evidence, code, frozen tables, figures, and analysis results are unchanged from v1.0.0. The update standardizes the author's Romanized name to **YeChan Jo** in repository citation metadata and documentation before Zenodo deposition.

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

## Public data policy

The release does **not** redistribute UCI HAR or WISDM source observations.

- UCI HAR: https://doi.org/10.24432/C54S4K
- WISDM: https://doi.org/10.24432/C5HK59

## Verification

```bash
python scripts/verify_public_results.py
```

The manuscript-facing evidence freeze is documented in `EVIDENCE_FREEZE_SUMMARY.md`, `DERIVED_TABLE_METHODS.md`, and `provenance/`.

## Zenodo

This v1.0.1 release is the version intended for Zenodo archival and DOI citation.
