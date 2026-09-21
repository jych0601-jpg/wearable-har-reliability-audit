# v1.0.0 release checklist

This checklist defines the final repository state to archive on Zenodo for the manuscript
**Evaluation Protocol Changes Probability Reliability for New-User Wearable Human Activity Recognition**.

## Scientific evidence

- [x] Frozen V1 primary source is present in `src/`.
- [x] Secondary-expansion scientific source is present in `src_expansion/`.
- [x] Fixed configs are present in `configs/expanded.json` and `configs/temporal_blocking.json`.
- [x] Compact headline result tables are present in `results/`.
- [x] Manuscript-level frozen tables are present in `tables/frozen/`.
- [x] Secondary/post-hoc descriptive tables are present in `tables/derived/`.
- [x] Six publication figures and figure manifest are present in `figures/`.
- [x] Evidence-freeze provenance and hash checks are present.
- [x] Primary/secondary/post-hoc interpretation boundaries are documented.

## Public-scope checks

- [x] Third-party raw datasets are not redistributed.
- [x] Internal ~23 GB archive is not published.
- [x] Model checkpoints and full raw predictions are excluded.
- [x] Local Windows/OneDrive paths and recovery metadata are excluded.
- [x] Repository-authored code/documentation license is present.
- [x] Dataset source DOIs are documented.

## Documentation

- [x] README describes current source/evidence structure.
- [x] REPRODUCIBILITY.md references the actual public config paths.
- [x] RESULT_PROVENANCE.md reflects that exact scientific source is already included.
- [x] CITATION.cff is present.
- [ ] Update CITATION.cff with v1.0.0 release date and Zenodo DOI after archival.

## Final pre-release actions

- [x] Confirm the manuscript's numerical claims still match the frozen public evidence.
- [x] Run `python scripts/verify_public_results.py` in a clean environment.
- [x] Confirm no accidental private/local files were added since the evidence freeze.
- [ ] Create GitHub release/tag `v1.0.0`.
- [ ] Archive `v1.0.0` on Zenodo.
- [ ] Record the version DOI in `CITATION.cff`.
- [ ] Insert the Zenodo DOI into the manuscript Code availability / Data Availability text as appropriate.
- [ ] Perform final manuscript render and submission audit.

Do not alter frozen result values after the archival release without issuing a new repository version and documenting the change.
