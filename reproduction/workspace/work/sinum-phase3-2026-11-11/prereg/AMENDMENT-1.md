# Amendment 1 to the Phase 3 pre-registration (2026-11-11T10:08Z)

Written before any NC, C or T outcome of the two new cohorts was examined. Seen so far: ten S outcomes of
cohort B (campaign log) and the number of completed cohort-A runs. The original files and their hashes
(prereg.sha256) are unchanged.

## Reason

The parallel Phase 1 session reports a user decision of 11 November (plan option A): Algorithm 1 now
optimizes every pair in the chart t = h^2 >= 0 (the former control T); discrete completion remains for
joined blocks of orders 2 and 3, with an inward-slope test; the former discrete pair completion
("old C", solver method 'adaptive') becomes an ablation. NC and S are unchanged.

## Changes to the analysis

* Cohort B (first-order PB, pair-only solver): the method C of the paper is the T code path
  (squared_gap.install). The primary comparison becomes T versus NC (pre-registered as the secondary
  pair NC-T), with the pre-registered tests: exact McNemar at 1e-6 and Wilcoxon on paired log10 errors,
  pooled and per dimension, Holm within the primary family. Old C is reported as an ablation with the
  same tests against NC and T. S comparisons stay secondary.
* Cohort A (1D independent recovery): the paper's C needs the Phase-1 chain solver. When that solver is
  validated, it is run on the same 200 starts (orders 2, 3; seeds 900-999), and new C versus NC becomes
  the primary comparison (McNemar at 1e-6 and 1e-10, Wilcoxon, per order). Old C versus NC is reported as
  the ablation. If the new-C runs are unavailable, the old-C comparison is reported and labelled as the
  ablation, not as the method.
* Thresholds, strata, effect measures, cost measure and retention rules are unchanged.

## Changes to scheduling (no effect on results)

* At most four Phase-3 worker processes (coordination with Phase 1; the host is memory-constrained).
* Cohort-B task order T, NC, C, S. Runs in progress at the switch were discarded and rerun; completed
  outputs were kept unchanged.
* Controlled timing runs in one idle window shared with Phase 1, after both accuracy campaigns finish,
  for NC and the paper's C (and the old-C ablation in first-order PB).
