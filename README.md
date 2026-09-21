# Wearable HAR Reliability Audit

Public reproducibility materials for the manuscript:

**Evaluation Protocol Changes Probability Reliability for New-User Wearable Human Activity Recognition**

Author: **YeChan Jo**  
Affiliation: Department of Artificial Intelligence, Gachon University

## Study question

This study audits whether the evaluation protocol itself changes conclusions about predictive reliability when wearable human activity recognition (HAR) models are deployed to unseen users.

The core comparison is:

- **sample-mixed evaluation**
- **subject-disjoint evaluation**

The study evaluates discrimination, proper probability scores, calibration transfer, confidence-based selective prediction, adaptive prediction sets, model-rank changes, and protocol-related confidence shifts.

## Public datasets

This repository does **not** redistribute source observations.

- UCI HAR: https://doi.org/10.24432/C54S4K
- WISDM Smartphone and Smartwatch Activity and Biometrics: https://doi.org/10.24432/C5HK59

Please obtain the datasets from their official repositories and respect their original terms.

## Analysis tiers

### Pre-specified primary analysis

- 2 public datasets
- 4 classical probabilistic classifiers
- primary seed: **20260911**
- 5 rotating folds
- sample-mixed vs subject-disjoint evaluation
- scalar temperature scaling
- confidence-based deferral
- participant-level inference
- 5,000 participant-bootstrap draws
- 5,000 Monte Carlo sign-flip draws
- Holm correction across four primary estimands

### Secondary robustness expansion

- fixed 1D-CNN sensitivity
- 10 fixed split seeds
- one-vs-rest isotonic calibration
- adaptive prediction sets (APS)
- UCI HAR overlap-reduced sensitivity analysis

Repeated seeds are sensitivity repetitions and are not treated as independent participants.

### Post-hoc / descriptive manuscript analyses

The manuscript evidence freeze also contains clearly labelled secondary descriptive analyses of:

- signed confidence gap (mean confidence minus accuracy),
- protocol-related model-rank changes,
- arithmetic decomposition of the UCI primary-seed contrast into an overlap-blocking step and a residual subject-disjoint step.

These analyses are not added retrospectively to the original primary hypothesis family.

## Headline findings

The pre-specified primary analysis found:

- subject-disjoint minus sample-mixed macro-F1: **-0.1092**
- subject-disjoint minus sample-mixed multiclass Brier score: **+0.1380**
- temperature-scaled minus raw Brier score under subject-disjoint evaluation: **+0.0077**
- retained error at nominal 80% coverage minus full error: **-0.0564**

The 10-seed, five-model secondary expansion preserved the same protocol-effect direction:

- macro-F1: **-0.1270**
- Brier: **+0.1572**
- NLL: **+0.5459**
- ECE: **+0.0424**

See `results/`, `tables/`, and `EVIDENCE_FREEZE_SUMMARY.md` for the manuscript-facing numerical evidence.

## Important interpretation

This repository supports an **evaluation-protocol audit**. It does not claim:

- a new classifier,
- a new uncertainty algorithm,
- universal calibration superiority,
- participant-specific conformal coverage guarantees under user shift,
- a causal decomposition of the protocol gap,
- or clinical safety.

UCI HAR uses 50% overlapping windows. The study therefore includes an overlap-reduced sensitivity analysis and interprets the main comparison as **protocol optimism**, which can combine represented-user similarity and temporal/window dependence.

For WISDM, the CNN used raw temporal windows while the classical models used the transformed ARFF representation. The deep-model extension is therefore a representation sensitivity rather than a paired-window comparison.

## Repository layout

- `src/` — exact frozen V1 primary-analysis source
- `src_expansion/` — exact scientific source used for the secondary expansion
- `configs/expanded.json` — fixed expanded-analysis configuration
- `configs/temporal_blocking.json` — UCI overlap-blocking configuration
- `results/` — compact headline result tables
- `tables/frozen/` — manuscript-level tables copied from the completed analysis
- `tables/derived/` — explicitly labelled secondary/post-hoc descriptive tables
- `figures/` — six frozen publication figures in PNG/PDF plus figure manifest
- `provenance/` — completed-analysis metadata and hash-verification records
- `EVIDENCE_FREEZE_SUMMARY.md` — manuscript evidence-freeze summary
- `DERIVED_TABLE_METHODS.md` — derivation rules for post-hoc descriptive tables
- `scripts/verify_public_results.py` — compact numerical verification
- `docs/` — dataset, methods, source-scope, provenance and reproducibility notes

## Repository scope

This public repository intentionally excludes:

- the original third-party datasets,
- the internal ~23 GB research archive,
- model checkpoints,
- the complete raw prediction archive,
- interrupted-run logs,
- local Windows paths,
- checkpoint/recovery metadata,
- and research-internal operational files not needed for publication-level verification.

The exact scientific source is included; the large internal archive remains the authoritative record for complete fit-level artifacts.

## Evidence freeze

The manuscript-facing evidence was frozen on **2026-09-14**. The evidence-freeze package includes completed manuscript tables, six figures, participant-level provenance used for the derived descriptive analyses, and SHA-256 verification records.

No model retraining was performed to create the confidence-gap, model-rank, or overlap-decomposition descriptive tables.

## Verify the public numeric artifacts

```bash
python scripts/verify_public_results.py
```

This script checks compact headline values. See the evidence-freeze manifests for the broader manuscript-facing artifact hashes.

## Release status

The repository is currently in **pre-release preparation**. A `v1.0.0` GitHub release will be created after final repository checks and then archived on Zenodo. The resulting persistent DOI will be added to the manuscript and `CITATION.cff`.

## License

Repository-authored code and documentation are released under the MIT License. The original UCI HAR and WISDM datasets are not included and remain subject to their original terms.

## Citation

Citation metadata are provided in `CITATION.cff`. A Zenodo DOI will be added after the `v1.0.0` release is archived.
