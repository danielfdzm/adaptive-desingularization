# Reproducing the paper

The repository includes the experiments reported in the current main paper and
supplement, their recorded outcomes, and the code used to analyze them. Start
with the [experiment map](../experiments/README.md) to find a particular result.
The [machine-readable catalog](../experiments/catalog.json) connects every
experiment to its records and scripts.

## Verify and browse

From the repository root, with Python 3.13 or 3.14:

```sh
python3 reproduce.py check
python3 reproduce.py list
python3 reproduce.py list EXPERIMENT_ID
```

These commands need only Python's standard library. `check` verifies the full
SHA-256 inventory and all catalog paths. It fails for missing, altered, or
unlisted files. Local environments, Python caches, `.git/`, and `outputs/` are
excluded. The manifest detects accidental changes; it is not a digital signature.

The check also rejects manuscript directories and sources, complete TeX
documents, bundled archives, and PDFs outside the catalogued figures, even if
their checksums have been added to the manifest. Numerical table fragments and
the experiments' NumPy reference arrays remain part of the distribution.

`list EXPERIMENT_ID` shows the paper references, records, and entry points for
one experiment. With no command, `reproduce.py` prints help.

## Install the scientific dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin the versions used by the first-order PB, variance, and
publication-figure environment. Some historical directional runs used Python
3.13.9 and NumPy 2.4.2; others used Python 3.14.0 and NumPy 2.3.5. The original
records retain their environment and source hashes. A fresh optimization can
vary with the interpreter, numerical libraries, and hardware; saved records are
the source for the published tables. Recorded wall times describe their original
shared machine and are not portable performance benchmarks.

## Rebuild the figures

```sh
python reproduce.py figures main
python reproduce.py figures supplement
# Or rebuild all fourteen publication figures:
python reproduce.py figures
```

The main figure selectors are `pb-solution`, `algorithm`, and
`forcing-solutions`. Supplement selectors are `local`, `recovery-tradeoff`,
`budget`, and `supplementary` (the eight remaining supplement figures).

```sh
python reproduce.py figures algorithm
```

Generated PDF, SVG, PNG, numerical samples, and fresh provenance/layout checks
are written to `outputs/figures/`. The PB field figure uses the manuscript's
881 × 881 sampling grid and 600-dpi export, so this is the largest figure job.
The forcing-solution builder recomputes an independent boundary-value reference
for evaluation. Neither job reruns optimization of the Gaussian models.

The published figure files remain in [figures/](../figures/). PDF metadata and
rendering can differ across Matplotlib/platform versions; numerical input records
are identified by their checksums.

## Rebuild tables and summaries

```sh
python reproduce.py tables EXPERIMENT_ID
# Run all catalogued saved-record analysis steps:
python reproduce.py tables
```

The wrapper runs the catalog's explicit `saved_records` steps in their listed
order, once per distinct script and argument list. It does not discover scripts
by filename or silently skip missing ones. Table fragments, summaries, and other
generated analysis files appear under `outputs/tables/workspace/`, preserving
their relative paths. Console output is saved in `outputs/tables/logs/`;
some historical summaries report rows only through that output. These are the
numerical rows/fragments from the records; the manuscript supplies their final
typesetting and numbering.

For the current main-paper Table 1, `tables independent-recovery` includes
`main_recovery_table.py`. It reads all 700 paired NC/C raw endpoints, verifies
the 100-start and 50-start cohort sizes, and exports the five rows as CSV,
Markdown, LaTeX and JSON with input hashes. The outputs are named
`table-main-recovery.*` under
`outputs/tables/workspace/work/sinum-phase3-2026-11-11/stats/`.

## How rebuilds protect the supplied records

Every rebuild verifies the release first, then copies `reproduction/workspace/`
to a temporary directory under `outputs/`. Scripts run only in that copy. The
wrapper exports their generated files and removes the temporary workspace.
Allow enough disk space for one additional workspace copy. A failed builder
stops the command and leaves a log identifying the error.

The internal directory names preserve the source snapshots' relative imports
and recorded provenance. The experiment map provides the reader-facing
organization. Original dated source names and references to earlier algorithm
numbering are retained as provenance; the experiment map uses the current paper.

## Numerical checks and fresh campaigns

Numerical validation and optimization entry points are listed separately in the
catalog and by `python reproduce.py list EXPERIMENT_ID`. They are never launched
by `check`, `figures`, or `tables`. They can be expensive, and some original
drivers write beside their input files. To run them, work in a separate copy of
`reproduction/workspace/` and follow that experiment's protocol and driver help.
Preserve all failures, work-limit returns, and explicitly reported pilot cohorts
when comparing a fresh campaign with the paper.

The frozen higher-order campaign is already present in the curated workspace;
no separately downloaded ZIP or neighboring research directory is needed.

## Maintaining the repository

After intentional source or documentation changes, run the interface checks,
review the changed files, and update the release inventory:

```sh
python -m unittest discover -s tests -v
python tools/update_checksums.py
python reproduce.py check
```

The checksum updater records the new inventory; it does not validate scientific
results. Preserve the original run records and their source/environment metadata
when changing the surrounding documentation or analysis code. Generated files
under `outputs/` stay outside the release inventory.
