# Manuscript Evidence Freeze — 2026-09-14

## Freeze integrity

This package freezes the manuscript-facing evidence from the completed wearable HAR reliability audit.

- base fits: 1000
- optional overlap-sensitivity fits: 25
- fixed split seeds: 10
- participant-metric rows: 24450
- completed-analysis artifact hashes verified: 16/16
- current figure PNG/PDF hashes verified: 12/12

No new model fitting was performed for the derived tables in this freeze.

## Primary confirmatory evidence

- macro-F1, subject-disjoint minus sample-mixed: -0.1092 [-0.1360, -0.0862], Holm p=0.0007998
- Brier: +0.1380 [+0.1092, +0.1739], Holm p=0.0007998
- temperature minus raw Brier under subject-disjoint: +0.0077 [-0.0077, +0.0231], Holm p=0.3567
- nominal-80% temperature selective minus full error: -0.0564 [-0.0636, -0.0494], Holm p=0.0007998

## Expanded robustness

- macro-F1: -0.1270
- Brier: +0.1572
- NLL: +0.5459
- ECE: +0.0424

All four protocol effects retain the same direction across the ten fixed seeds. These are secondary robustness results. The broad expanded family contains 270 planned/testable contrasts; with 5,000 sign-flip draws the minimum attainable Holm-adjusted p is about 0.053989, so the expanded family should not be described as conventionally significant at p<0.05.

Calibration/uncertainty results:
- temperature minus raw Brier: +0.0110
- isotonic minus raw Brier: -0.0103
- isotonic minus raw ECE: -0.0237
- raw selective 80% retained-minus-full error: -0.0565
- APS mean set size: 3.56 at nominal 80%, 5.07 at nominal 90%

## Secondary descriptive confidence gap

`confidence_gap = mean confidence - accuracy`; positive values indicate overconfidence.

Raw-probability, seed-averaged estimates:
- UCI HAR shift, subject-disjoint minus mixed: +0.0146 [+0.0015, +0.0295]
- WISDM shift: +0.0967 [+0.0654, +0.1334]
- equal-dataset pooled shift: +0.0556 [+0.0381, +0.0760]

The pooled gap moves from -0.0545 under sample mixing to +0.0011 under subject-disjoint evaluation.

Interpretation: the signed confidence gap moves in the overconfidence direction on average, especially on WISDM, but not for every UCI model. This analysis is post-hoc secondary/descriptive, uses pointwise participant-bootstrap intervals, and is not added to the original primary hypothesis family.

## UCI overlap decomposition

For raw UCI predictions at fixed seed 20260911, pooled equally over five models:

- macro-F1: overlap-blocking step -0.0242; residual subject-disjoint step -0.0249; total -0.0491
- Brier: overlap-blocking step +0.0371; residual +0.0351; total +0.0722
- NLL: overlap-blocking step +0.0836; residual +0.0926
- ECE: overlap-blocking step +0.0142; residual +0.0163

For macro-F1/Brier/NLL/ECE, the arithmetic decomposition is roughly half overlap-blocking step and half residual subject-disjoint step. This is not a causal explained-percentage claim.

## Descriptive rank changes

Top raw model under sample-mixed -> subject-disjoint:
- uci_har, macro_f1: hist_gradient_boosting -> logistic
- uci_har, brier: hist_gradient_boosting -> logistic
- uci_har, nll: hist_gradient_boosting -> logistic
- uci_har, ece_15: hist_gradient_boosting -> logistic
- wisdm_watch_accel, macro_f1: hist_gradient_boosting -> hist_gradient_boosting
- wisdm_watch_accel, brier: hist_gradient_boosting -> random_forest
- wisdm_watch_accel, nll: hist_gradient_boosting -> hist_gradient_boosting
- wisdm_watch_accel, ece_15: hist_gradient_boosting -> random_forest

Rank changes are descriptive. The derived CSV records Spearman rank correlations for all five models and for the four classical models separately. CNN/classical representation differences must remain explicit, especially for WISDM.

## Figure freeze

The six current figures and their PDF/PNG versions are frozen:
1. protocol effects
2. calibration
3. selective risk-coverage
4. APS coverage/set size
5. seed sensitivity
6. temporal-overlap sensitivity

## Manuscript claim guardrails

Safe:
- sample-mixed evaluation can be optimistic for unseen-user discrimination and probability reliability;
- protocol-effect direction is stable across ten fixed seeds;
- temperature scaling does not consistently transfer;
- isotonic shows favorable Brier/ECE signals but not conventional significance after the broad Holm family;
- confidence-based deferral reduces retained-set error;
- signed confidence gap moves in the overconfidence direction on average, with model/dataset heterogeneity;
- exact-window overlap contributes materially but does not exhaust the UCI primary-seed protocol gap.

Avoid:
- claiming the expanded 270-contrast family is p<0.05 significant;
- claiming every model becomes more overconfident;
- claiming APS has a participant-shift coverage guarantee;
- calling the overlap arithmetic ratio a causal explained fraction.
