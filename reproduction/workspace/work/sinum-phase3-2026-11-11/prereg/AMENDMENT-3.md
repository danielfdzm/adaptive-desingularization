# Amendment 3 to the Phase 3 pre-registration (2026-11-11T12:41Z)

## Reason

In cohort B the method C (the T chart) evaluates pairs with the exact parity formula of squared_gap.py,
while the frozen NC evaluates them with the eight-node averaged derivative rule. The Phase-1 session found
that this evaluator difference changes NC's outcome at the 1e-6 threshold: with the exact evaluator in the
same h-chart (NC-exact), NC reached 1e-6 in 11 of the first 18 control runs (median 5.4e-7), against about
6 of 20 with the averaged rule. The comparison C versus frozen NC therefore mixes the chart (h vs t) with
the evaluator.

Seen before this amendment: all cohort-B outcomes of NC, C, C_d and the first 42 S runs (the primary
analysis of amendment 1 was already computed), and the Phase-1 interim aggregate above. This amendment is
therefore not blind to the C-versus-frozen-NC result; it adds a control and changes which comparison is
called primary, and both comparisons are reported.

## Changes

* New arm NC-exact on all 150 cohort-B starts: numerical-section-revision/phase1-2026-11-11/
  first_order_runner.py --mode pb_NC_analytic (frozen solver, method 'stable', h-chart, exact parity
  evaluator), sha256 b90dd58cabb095d00aaa7b178bb28cc0ac96c7f6342cb3a2a50237114734a993.
  Output: cohort_pb/results_nc_exact/. Seeds 10-14 must reproduce the Phase-1 evaluator-control records.
* Primary cohort-B comparison: C versus NC-exact (same evaluator; a pure chart comparison), with the
  pre-registered tests (exact McNemar at 1e-6, Wilcoxon on paired log10 errors, pooled and per dimension,
  Holm within the family). C versus frozen NC is reported as well, with the confound stated. Fits are
  compared both ways.
* Scheduling: the S arm pauses while NC-exact runs; completed S outputs are kept.

## Addendum (2026-11-11T12:55Z)

Agreed with the Phase-1 session: the ablation C_d is rerun as well with the exact parity evaluator
(first_order_runner.py --mode pb_Cd_analytic, same runner and hash) on all 150 cohort-B starts, output
cohort_pb/results_cd_exact/ (suffix _Cdx), so that S, NC, C and C_d in the main-text first-order rows
share one pair evaluator (S has no pairs). The averaged-evaluator NC and C_d results move to SM1. No
C_d-exact outcome has been seen; there is no Phase-1 record to cross-check it against.
