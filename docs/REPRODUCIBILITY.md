# Reproducibility notes

## What can be verified immediately

Run:

```bash
python scripts/verify_public_results.py
```

This checks that the compact public CSVs contain the manuscript-level primary and
expanded estimates recorded in the audited research report.

## Full rerun requirements

A full rerun requires:

1. official UCI HAR and WISDM source data,
2. the exact cleaned fitting/calibration/statistics source files from the archived
   research package,
3. the fixed configuration and seeds recorded in `configs/study_public.json`.

The exact scientific source is intentionally not re-created from prose. It should be
published from the archived source tree before the Zenodo v1.0.0 release.
