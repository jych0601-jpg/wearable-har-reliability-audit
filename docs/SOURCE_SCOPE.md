# Public source scope

## Included exact expansion source

The following files are copied from the completed research package without reconstructing
their scientific logic:

- `src_expansion/prepare_data.py`
- `src_expansion/deep_har.py`
- `src_expansion/reliability.py`
- `src_expansion/experiments.py`
- `src_expansion/postprocess.py`
- `src_expansion/temporal.py`
- `src_expansion/supplemental_summaries.py`

## Deliberately excluded internal tooling

The public repository omits recovery scripts, pause/resume bookkeeping, renderer setup,
Word-manuscript generation, archive sealing, local environment repair scripts, `.pyc` files,
and other research-internal operational tooling.

## Frozen primary-analysis source

The exact V1 source modules from the archived primary-analysis snapshot are included in `src/`.
They were copied directly from the frozen archive and were not reconstructed from the manuscript.

Published scientific modules include the dataset loaders, split construction, fixed classical
estimators, temperature calibration, metrics, experiment execution, participant-level analysis,
run audit, plotting and reproduction helpers.
