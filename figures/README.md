# Publication figures

These are exactly the figures included by the supplied paper and supplement.
The current plotting masters read saved records; the plotting commands do not rerun optimization.

| Paper location | Figure | Plotting source |
| --- | --- | --- |
| Figure 1 | [PB solution and pointwise errors](figure_pb_solution_maps.pdf) | [build_pb_solution_maps.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_pb_solution_maps.py) |
| Figure 2 | [Projected pair methods in t and h](figure_alg41_cohort.pdf) | [build_algorithm.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_algorithm.py) |
| Figure 3 | [Solutions for prescribed forcing](figure_forcing_solutions.pdf) | [build_forcing_solutions.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_forcing_solutions.py) |
| Figure SM1 | [Local pair geometry and tolerance diagnostics](figure_local.pdf) | [build_local.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_local.py) |
| Figure SM2 | [Paired recovery accuracy and work](figure_recovery_tradeoff.pdf) | [build_recovery_tradeoff.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_recovery_tradeoff.py) |
| Figure SM3 | [Matched NC/C recovery trajectories](figure_recovery.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM4 | [Independent and PB final-error distributions](figure_ecdf.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM5 | [PB final-error distributions by dimension](figure_ecdf_pb_dimensions.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM6 | [Prescribed-forcing budget study](figure_budget.pdf) | [build_budget.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_budget.py) |
| Figure SM7 | [One-dimensional solution recovery](figure_solution_1d.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM8 | [Higher-order PB recovery](figure_order.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM9 | [PB solution and errors with projected centers](figure_solution_pb.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM10 | [Gaussian-variance family](figure_variance_family.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| Figure SM11 | [First-order PB recovery across dimensions](figure_dimension.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |

Run `python reproduce.py figures` from the repository root to rebuild all 14 figures.
See the [reproduction guide](../docs/REPRODUCING.md) for individual builders and output locations.

The plots retain the manuscript’s fixed colors and panel geometry. Input hashes,
layout audits, and numerical checks are stored with the plotting masters. The
main Figure 1 uses the manuscript’s plasma solution colormap and blue–white–red
error colormap, shared limits, an 881 × 881 sampling grid, and 600 dpi export.
