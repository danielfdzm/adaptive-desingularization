# Amendment 2 to the Phase 3 pre-registration (2026-11-11T11:08Z)

Written before any run under this amendment. All earlier records are kept unchanged.

## Reason

In the one-dimensional cohort, 10 of 200 runs of the method C (tpair) ended because the coefficient
system was singular at a single L-BFGS-B *trial* point (a line-search step to the box corner placed an
atom on a pair center), although the accepted state was regular. The frozen solvers end the whole run in
this case. Algorithm 1 states that failed eligibility checks retain the previous state, so the Phase-1
session (owner of the method) changed the solver: a failed coefficient solve at a trial point now
returns +inf to the line search, which backtracks, as for a guard-rejected trial. A failure at an
accepted state still ends the run. The rule applies to every method (S, NC, C_d, C).

## Changes

* Solver: numerical-section-revision/phase1-2026-11-11/src/chain_solver.py (Solver.evaluate_trial),
  hash recorded in each run record and in the campaign protocol.
* Cohort A is rerun for all four methods (S standard_restart, NC stable, C_d adaptive, C tpair) on the
  same 200 starts (orders 2, 3; seeds 900-999), same problem, fit limit, stopping rule and tests.
  Results go to cohort1d/results_amend2/; the pre-amendment records stay in cohort1d/results/.
* Check: every run without a failed trial must reproduce its pre-amendment record exactly.
* Primary analysis: the amendment-2 cohort. The pre-amendment counts of aborted runs (C 10, NC 3,
  C_d 3, S 34 of 200) and the pre-amendment results are reported in the supplement.
* Cohort B is unaffected: no NC, C_d or C run there ended on a coefficient-solve failure.
* Stationarity rows are regenerated with the updated cleanup module (backtracking at failed trials).
