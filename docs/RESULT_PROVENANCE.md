# Result provenance

## Completed analysis record

The completed research package preserved:

- archived V1 primary predictions and exact recomputation of the primary evidence,
- 800 core classical-model fits,
- 200 core CNN fits,
- 25 UCI overlap-reduced sensitivity fits,
- calibration and conformal outputs,
- participant-level analysis tables,
- integrity manifests and validation tests.

The internal research archive remains the authoritative complete record for these fit-level artifacts.

## Public manuscript-facing evidence

The public repository contains two complementary result layers.

### Compact headline results

The `results/` directory exposes the principal manuscript claims in a compact, human-readable form.

### Frozen manuscript evidence

The manuscript evidence freeze, created on **2026-09-14**, contains:

- `tables/frozen/` — completed manuscript-level tables copied from the finished analysis,
- `figures/` — six frozen publication figures in PNG and PDF form,
- `provenance/` — completed-analysis metadata plus figure/analysis hash verification,
- `EVIDENCE_FREEZE_SUMMARY.md` — the manuscript-facing evidence summary.

The evidence-freeze hash checks were performed against the completed analysis artifacts before manuscript revision.

## Derived descriptive tables

The `tables/derived/` directory contains explicitly labelled post-hoc / secondary descriptive analyses derived from already completed participant-level outputs.

These include:

- confidence-gap protocol comparisons,
- protocol-related model-rank changes,
- the UCI primary-seed overlap/residual arithmetic decomposition.

No model retraining was performed to create these tables. Their derivation rules are documented in `DERIVED_TABLE_METHODS.md`.

These descriptive analyses are not retroactively added to the original primary hypothesis family and do not replace the frozen confirmatory primary evidence.

## Scientific source provenance

The exact scientific code used in the research package is now included:

- `src/` for the frozen V1 primary analysis,
- `src_expansion/` for the secondary expansion.

Older documentation stating that source files still needed to be copied before Zenodo archival was stale and is superseded by this document.

## Deliberate exclusions

The public repository does not reproduce the complete internal ~23 GB archive. It intentionally excludes:

- third-party raw datasets,
- model checkpoints,
- the full raw-prediction archive,
- interrupted-run logs,
- local recovery/checkpoint state,
- and machine-specific paths.

This keeps the public release reviewable while preserving the manuscript-facing scientific source, fixed configurations, result tables, figures and provenance.

## Release provenance

The current repository is a pre-release snapshot. After final checks, `v1.0.0` will define the archival manuscript-supporting repository state. That release will be deposited in Zenodo, and its DOI will be recorded in the manuscript and citation metadata.
