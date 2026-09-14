# Methods summary

## Partitioning

Five rotating folds were used. For fold `k`, fold `k` was the final test partition,
fold `(k + 1) mod 5` was the calibration partition, and the remaining three folds
were used for model training.

Under the **subject-disjoint** protocol, participant identities were separated across
training, calibration, and test partitions. Under the **sample-mixed** comparator,
individual windows were stratified across folds, allowing the same participant to appear
in multiple roles.

## Primary models

- multinomial logistic regression
- random forest
- Extra Trees
- histogram gradient boosting

## Secondary deep sensitivity

A fixed 1D-CNN was added in the secondary expansion. The archived study report records:
64/128 convolutional channels, kernel size 5, dense layer 128, dropout 0.3, Adam
learning rate 0.001, weight decay 0.0001, batch size 128, maximum 40 epochs, and
validation-NLL patience 6.

The CNN uses raw temporal windows. The classical WISDM analysis uses transformed ARFF
features; these representations are not row-aligned.

## Calibration and uncertainty

- scalar temperature scaling
- one-vs-rest isotonic calibration with renormalization
- confidence-quantile selective prediction
- adaptive prediction sets (APS), alpha 0.20 and 0.10

Calibration and final test participants are disjoint in the subject-disjoint protocol.

## Statistical inference

Participants are the inferential units. The primary analysis used:

- 5,000 participant-cluster bootstrap draws
- 5,000 Monte Carlo sign-flip draws
- Holm adjustment across four pre-specified primary estimands

The secondary 10-seed analysis averages repeated seeds within participant before
participant-level inference. A broad 270-contrast secondary family was retained; with
5,000 sign-flip draws the minimum attainable Holm-adjusted value was 270/5001 = 0.0539892.
