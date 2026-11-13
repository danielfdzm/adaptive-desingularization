# Reproduction workspace

Start with the [experiment index](../experiments/README.md) or the [reproduction guide](../docs/REPRODUCING.md).

`workspace/` contains the selected solver sources, protocols, complete reported cohorts, validation outputs, and figure/table builders. Its internal paths preserve the relative imports and dataset locations used by the recorded campaigns. This allows the original numerical code to remain traceable to the source hashes stored in the results. Folder dates identify campaigns, not competing versions of the paper.

The internal `workspace/paper_1111/` directory contains validation and plotting
code only. It does not contain the paper or supplementary manuscript.

`selection.json` records the original source path, SHA-256, and inclusion reason for every selected workspace file. The new main-table
exporter is explicitly marked as newly authored code. The root `SHA256SUMS` verifies the final distribution. Saved runs are retained byte for byte; selected frozen solver snapshots are kept where the experiments depend on them. Numerical reruns may differ across dependency versions and hardware.

Publication figures are collected in [`../figures/`](../figures/). The supported commands rebuild in a separate output directory, preserving the supplied records and plots.
