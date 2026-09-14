# Wearable HAR Reliability Audit

Minimal public reproducibility materials for the manuscript:

**Evaluation Protocol Changes Probability Reliability for New-User Wearable Human Activity Recognition**

Author: **Yechan Cho**  
Affiliation: Department of Artificial Intelligence, Gachon University

## Study question

This study audits whether the evaluation protocol itself changes conclusions about predictive reliability when wearable human activity recognition (HAR) models are deployed to unseen users.

The core comparison is:

- **sample-mixed evaluation**
- **subject-disjoint evaluation**

The study evaluates discrimination, proper probability scores, calibration transfer, confidence-based selective prediction, and adaptive prediction sets.

## Public datasets

This repository does **not** redistribute source observations.

- UCI HAR: https://doi.org/10.24432/C54S4K
- WISDM Smartphone and Smartwatch Activity and Biometrics: https://doi.org/10.24432/C5HK59

Please obtain the datasets from their official repositories and respect their original terms.

## Analysis tiers

### Pre-specified primary analysis

- 2 public datasets
- 4 classical probabilistic classifiers
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

See `results/` for the auditable numeric tables.

## Important interpretation

This repository supports an **evaluation-protocol audit**. It does not claim:

- a new classifier,
- a new uncertainty algorithm,
- universal calibration superiority,
- participant-specific conformal coverage guarantees under user shift,
- or clinical safety.

UCI HAR uses 50% overlapping windows. The study therefore includes an overlap-reduced sensitivity analysis and interprets the main comparison as **protocol optimism**, which can combine represented-user similarity and temporal/window dependence.

For WISDM, the CNN used raw temporal windows while the classical models used the transformed ARFF representation. The deep-model extension is therefore a representation sensitivity rather than a paired-window comparison.

## Repository scope

This public repository intentionally excludes:

- the original third-party datasets,
- the internal 23 GB research archive,
- model checkpoints,
- interrupted-run logs,
- local Windows paths,
- checkpoint/recovery metadata,
- and other research-internal files not needed for publication-level verification.

## Current public package status

The current public package contains the fixed analysis metadata, headline processed results, and verification scripts.

The repository now includes the exact scientific source used for the secondary expansion in `src_expansion/`:
data restoration, fixed 1D-CNN training, model-fit orchestration, isotonic calibration, APS,
expanded statistics/figures, temporal-overlap sensitivity, and descriptive supplements.

The repository includes the exact frozen V1 baseline source in `src/` and the exact
secondary-expansion scientific source in `src_expansion/`. The baseline modules were copied
from the archived primary-analysis snapshot rather than reconstructed from the manuscript.

This distinction prevents publishing code that merely approximates the audited implementation.

## Verify the public numeric artifacts

```bash
python scripts/verify_public_results.py
```

## License

Repository-authored code and documentation are released under the MIT License. The original UCI HAR and WISDM datasets are not included and remain subject to their original terms.

## Citation

Citation metadata are provided in `CITATION.cff`. A Zenodo DOI will be added after the public repository is finalized and archived.
