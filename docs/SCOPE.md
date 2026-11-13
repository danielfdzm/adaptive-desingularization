# What belongs in this repository

The inclusion boundary is the November 12, 2026 version of the main paper
and its supplement. This repository distributes their experiments, numerical
records, figures, and reproduction instructions; the manuscript PDFs and
LaTeX sources are excluded.

## Included

- Every experimental family reported in either document, linked in the
  [experiment index](../experiments/README.md).
- The full reported cohorts, including failed runs, interrupted or capped
  outcomes, pre-amendment results, and explicitly discussed pilots.
- Historical comparisons when the current supplement reports them, such
  as evaluator-matched controls and discrete pair completion.
- Protocols, numerical validation, reference solutions, and the solver
  dependencies needed to interpret or reproduce those experiments.
- The current plotting masters and exactly the 14 figures included by
  the two documents.

## Excluded

- The main paper and supplementary manuscript, in PDF or LaTeX form.

- The superseded October six-figure/six-table study, including its
  certificate and budget-sweep experiments.
- The unreported hierarchical-jet and normal-family numerical studies.
- Unreported development tests, smoke campaigns, and launch diagnostics.
- Manuscript backups, editorial patch scripts, referee reports, and
  obsolete manuscript or report bundles.
- Unused publication figures, including the earlier reversible-pair
  editorial replot.
- Duplicate repository snapshots, ZIP archives, caches, and build debris.

Dependencies are distinguished from experiments: a Gaussian evaluator or
frozen solver may be needed by a reported study even when its original
standalone experiment is excluded. Required source snapshots retain their
original locations inside the reproduction workspace. They are not
additional advertised studies.

## Traceability

[`reproduction/selection.json`](../reproduction/selection.json) records the
original workspace paths, inclusion reasons, and hashes of the selected
files. The distribution-wide `SHA256SUMS` verifies the delivered files.
The machine-readable [catalog](../experiments/catalog.json) provides a
separate, explicit list of the reported experiments. Rebuilding figures
uses the current publication builders; older campaign-local plots are not
publication masters.

Historical dates and algorithm numbers in frozen records are provenance.
Use the paper locations and method names in the experiment index to
interpret them. No measured outcome has been changed during organization.

The public repository starts with a clean history containing only this
experiment package. Earlier commits and release tags containing manuscripts
remain in a separate private archive; they are not part of the public repository.
