# Experiment and analysis tools

See [the reproduction guide](../docs/REPRODUCE.md) for the paper-table mapping.
Run `python -m ablations.run --list` to inspect all available conditions.
Configurations are in `config/` at the repository root.

The three `incremental_*` conditions retain CRD, matching the manuscript's
"all other modules unchanged" statement for Table 2. The older single-objective
controls disable CRD and are separate scientific conditions.

Metrics, long-tail summaries, prediction transitions, and ordering-case analyses
read saved predictions/checkpoints. No private API clients or downloaded baseline
repositories are included.
