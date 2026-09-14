# Derived table methods

## Confidence gap
Source: `provenance/participant_metrics.csv`.
Raw probabilities only. Ten seeds are averaged within participant/model/protocol. Dataset summaries average five models within participant. The pooled summary gives equal weight to the two dataset means. Pointwise 95% intervals use 5000 participant-bootstrap draws with fixed derivation RNG seed 20260914. No p-values or multiplicity correction are added.

## Overlap decomposition
Source: `provenance/participant_metrics.csv`.
UCI HAR, raw probabilities, seed 20260911. No new fit is run. The arithmetic identity is:
`subject-disjoint - sample-mixed = (leakage-reduced-mixed - sample-mixed) + (subject-disjoint - leakage-reduced-mixed)`.
Pooled rows average five models within participant. Pointwise 95% intervals use 5000 participant-bootstrap draws. Fraction columns are arithmetic ratios only, not causal attribution.

## Rank changes
Source: frozen `table_expanded_model_ranks.csv`.
Raw probabilities, sample-mixed vs subject-disjoint, macro-F1/Brier/NLL/ECE. Descriptive only. Spearman correlations are provided for all five models and separately for the four classical models. Representation differences remain explicit.
