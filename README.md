# Wearable HAR Reliability Audit

Reproducibility materials for the study **“Evaluation Protocol Changes Probability Reliability for New-User Wearable Human Activity Recognition.”**

This project audits how evaluation protocol changes conclusions about discrimination and probability reliability in wearable human activity recognition (HAR), with emphasis on deployment to previously unseen users.

## Study scope

The frozen primary analysis compares **sample-mixed** and **subject-disjoint** evaluation on two public datasets using four fixed probabilistic classifiers. Secondary robustness analyses add a fixed 1D-CNN, isotonic calibration, adaptive prediction sets (APS), ten split seeds, and an overlap-reduced UCI HAR sensitivity analysis.

The contribution is an **evaluation-protocol reliability audit**. It does not claim a new classifier, a new calibration algorithm, or universal conformal coverage under participant shift.

## Public datasets

The original datasets are **not redistributed** in this repository.

- UCI HAR — Human Activity Recognition Using Smartphones: https://doi.org/10.24432/C54S4K
- WISDM Smartphone and Smartwatch Activity and Biometrics: https://doi.org/10.24432/C5HK59

Please obtain the datasets from their official repositories and comply with their original terms.

## Main reported findings

### Pre-specified primary analysis

- Subject-disjoint minus sample-mixed macro-F1: **−0.1092**
- Subject-disjoint minus sample-mixed multiclass Brier score: **+0.1380**
- Temperature-scaled minus raw Brier under subject-disjoint evaluation: **+0.0077**
- Selective error at nominal 80% retention minus full-coverage error: **−0.0564**

### Secondary robustness expansion

Across five models and ten fixed split seeds:

- Subject-disjoint minus sample-mixed macro-F1: **−0.1270**
- Brier score: **+0.1572**
- NLL: **+0.5459**
- ECE: **+0.0424**
- APS observed participant-mean coverage: **0.8080** at nominal 80% and **0.9049** at nominal 90%

The expanded 270-contrast family is reported as secondary robustness evidence. With 5,000 sign-flip draws, no pooled expanded contrast crossed the Holm-adjusted 0.05 threshold; this is not hidden or redefined post hoc.

## Repository contents

- `src/` — exact frozen primary-analysis Python source
- `tests/` — primary-analysis unit/regression tests
- `configs/` — frozen primary and sensitivity configurations
- `results/` — selected manuscript-facing primary and expanded result summaries
- `docs/` — dataset, methods, and provenance notes

The public repository intentionally excludes third-party raw datasets, model checkpoints, full raw predictions, local execution logs, recovery/checkpoint files, and the large internal research archive.

## Reproducibility notes

- Participants, not sensor windows, are the inferential units.
- Training, calibration, and test participants are pairwise disjoint in the new-user protocol.
- UCI HAR uses 50% overlapping windows; a secondary overlap-reduced sensitivity analysis shows that verified shared-window dependence explains part of naive sample-mixed optimism.
- WISDM classical models use the distributed transformed representation, whereas the CNN extension uses raw temporal windows. The deep-model result is therefore a representation sensitivity rather than a paired-window comparison.
- Calibration and APS are fitted without using final test labels.
- Standard conformal exchangeability is not assumed to hold at the participant level under unseen-user shift.

## Environment for the frozen primary analysis

- Python 3.11.15
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.17.1
- scikit-learn 1.9.0

See `requirements.txt` and `docs/REPRODUCIBILITY.md`.

## Citation

A `CITATION.cff` file is included. A Zenodo DOI will be added to the repository and manuscript after the public release is archived.

## License

Repository-authored code is released under the MIT License. The upstream UCI HAR and WISDM datasets are not included and remain subject to their original terms.
