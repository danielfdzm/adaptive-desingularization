# Phase 3 pre-registration (written 2026-11-11, 09:50 UTC, before any new run)

Plan: `work/sinum-revision-plan-2026-11-11/sinum-revision-plan.tex`, steps 10-12.
Nothing in this file may change after the first run of a cohort starts. The
SHA-256 of this file and of `prereg.json` is recorded in `prereg.sha256`
before the campaigns are launched.

## Cohort A: one-dimensional independent recovery (Table 1, panel A)

* Orders r = 2, 3. Seeds 900-999 (100 per order). Seeds 900-909 are the ten
  existing confirmation cases; their NC/C reruns must reproduce the guarded
  records exactly, and their S reruns the historical S records.
* Problem, exactly as in the existing study: sigma = 0.65, lambda = 10,
  P = r + 1, target a sigma^r G^(r)(x - b) with b ~ U[-0.6, 0.6] and
  a ~ U[0.7, 1.3] drawn by `default_rng(100000 + seed)`; P initial centers
  iid U[-1.2, 1.2] drawn by `default_rng(seed)`; training rule 384-node
  Gauss-Legendre on [-10, 10]; center box 3; fit limit 1200; max order 3.
* Methods: S = `standard_restart`, NC = `stable`, C = `adaptive` of the
  guarded directional solver (`guarded-rerun-2026-11-09/directional/src`),
  unchanged. All three stop after two local returns without a structural
  change, or at the fit limit.
* Endpoint error: 80-digit whole-line Gaussian inner products (solver summary).

## Cohort B: first-order Poisson-Boltzmann recovery (Table 2, rows r = 1)

* d = 5, 10, 20; geometries 0, 1; seeds 10-34 (25 per geometry, 50 per
  dimension, 150 starts). Seeds 10-14 are the existing starts; NC/C/T reruns
  must reproduce the existing records exactly.
* Problem unchanged: lambda = 1, sigma = 0.45, P = 10, target-informed
  starts of `config.start_centers`, QMC power 12, quadrature seed 8128,
  fit limit 1500, L-BFGS-B ftol = 1e-15, gtol = 1e-12, maxls 40.
* Methods:
  * NC = `stable`, C = `adaptive`: frozen guarded `run.py` (squared-gap/source,
    byte-identical to the guarded first-order source), unchanged;
  * T: the same solver with `squared_gap.install` (t = h^2 >= 0), unchanged;
  * S = `standard` ordinary Gaussians with a derived run loop whose only
    change is the stopping rule: two local returns (S has no structural
    changes), instead of the first local return. The ordinary-Gram condition
    stop (cond > 1e16) of S is kept.
* Endpoint error: 110-digit Gaussian inner products.

## Statistics (both cohorts)

* Strata: each order (A); each dimension and pooled over dimensions (B).
* Thresholds tau in {1e-4, 1e-6, 1e-8, 1e-10}; success means E <= tau.
* Primary comparison: C versus NC.
  * A: two-sided exact McNemar tests at tau = 1e-6 and 1e-10, and a two-sided
    Wilcoxon signed-rank test on paired log10 errors, for each order.
  * B: two-sided exact McNemar test at tau = 1e-6 and the Wilcoxon test on
    paired log10 errors, pooled over dimensions and per dimension.
  * Holm correction within the primary family of each cohort is reported
    alongside raw p-values.
* Secondary: all other thresholds; pairs C-S and NC-S (A, B); C-T and NC-T (B).
* Reported for each comparison: discordant counts, exact p-value, median of
  paired log10(E_1/E_2) with a distribution-free 95% interval (order
  statistics of the sign test), and the empirical distribution function of
  the final error for every method.
* Cost measure: coefficient fits (median per method; paired Wilcoxon on fits
  for C versus NC). Fits include line searches and completion trials.
* All runs are retained, including work-limit returns, abnormal returns and
  numerical failures.

## Controlled timing (supplement)

* The nine timed studies and their prescribed starts are unchanged from the
  existing timing table (first-order PB (0,10),(0,11),(1,10) per dimension;
  higher-order (0,10),(0,11),(1,10); full-span seeds 10-12; second-order 1D
  seeds 40-42; independent seeds 900-902 per order).
* Methods NC and C (and T for the first-order PB rows). Five serial
  repetitions per start and method, one numerical thread, no other numerical
  job running, method order alternating between repetitions.
* Reported: median over all repetitions and starts, and the range of the
  per-start medians. Every timed run must reproduce the endpoint and fit count
  of its accuracy run exactly.

## Step 10: tests of the local collision predictions

* Toy problems: the matched Gaussian-variance protocol (joint L-BFGS-B over
  coefficients and geometry, 3201-node trapezoid rule on [-16, 16], squared
  H1 error), sigma = 1, P = 2, target G_s' with variance s = 1 (zero gap,
  kappa = 0) and s = 0.99 (positive gap, kappa > 0), seeds 0-9.
  Charts: NC (half-gap h >= 1e-12), C (NC plus completion), T (t = h^2 >= 0).
* (a) ftol = 0 and gtol in {10^(-1), 10^(-1.5), ..., 10^(-14)} at s = 1, and
  for the first-order PB start (d, geometry, seed) = (5, 0, 10) with NC, C and
  T (fit limit raised to 20000 so that the tolerance, not the work limit,
  ends the run). Slopes of log10 E against log10 gtol are fitted over the
  runs whose last local return met the projected-gradient tolerance and
  whose error exceeds ten times the smallest error of that method.
  Predictions: 2/3 in h (NC), 1 in t (T) and after completion (C).
* (b) h_k against iteration for NC with ftol = gtol = 0 at s = 1 and s = 0.99.
  Prediction: geometric decay of h_k to the bound when kappa > 0; slow decay
  when kappa = 0. Ordinary weights grow like |c|/(2 h_k).
* (c) T at s = 0.99: the first iteration after which t_k = 0 holds until the
  end. Prediction: finite identification (strict complementarity).
