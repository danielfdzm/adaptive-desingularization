# Experiments in the paper

This catalog covers the current paper **and its supplement**. Every retained experiment is tied below to a reported result, including unsuccessful runs, negative controls, earlier protocol versions discussed in the supplement, and numerical validation. The dated source paths preserve the imports and provenance of the recorded runs.

Paper references below identify the November 12, 2026 manuscript version; the manuscript documents are not included. Method **S** uses ordinary Gaussians; **NC** uses stable coordinates without completion; **C** is adaptive completion; **Cd** is the discrete-completion ablation. The reversible projected pair method is a separate algorithm.

## Start with the main results

| Study | Reported result | Saved evidence |
| --- | --- | --- |
| [Independent second- and third-order recovery](#independent-recovery) | Table 1 | [results_amend2](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort1d/results_amend2) |
| [First-order Poisson–Boltzmann recovery in dimensions 5, 10 and 20](#pb-recovery) | Figure 1, Table 1 | [results](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/results) |
| [Reversible pair method and three comparators on the PB cohort](#reversible-pb) | Figure 2 | [results](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/results) |
| [Prescribed forcing with varying Gaussian budget](#forcing-budget) | Figure 3 | [runs](../reproduction/workspace/work/pstudy-forcing-2026-11-12/results/runs) |

## Paper figures

Only these 14 PDFs are used by the current manuscript sources. The publication builders replot saved records; solution-map builders also evaluate saved endpoints and independent reference solutions. They do not rerun optimization.

| Figure | Plot | Builder |
| --- | --- | --- |
| 1 | [PB solution and error maps](../figures/figure_pb_solution_maps.pdf) | [build_pb_solution_maps.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_pb_solution_maps.py) |
| 2 | [Projected methods on the PB cohort](../figures/figure_alg41_cohort.pdf) | [build_algorithm.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_algorithm.py) |
| 3 | [Prescribed-forcing solution curves](../figures/figure_forcing_solutions.pdf) | [build_forcing_solutions.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_forcing_solutions.py) |
| SM1 | [Local pair scaling and closing](../figures/figure_local.pdf) | [build_local.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_local.py) |
| SM2 | [Paired accuracy and work ratios](../figures/figure_recovery_tradeoff.pdf) | [build_recovery_tradeoff.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_recovery_tradeoff.py) |
| SM3 | [Linear and PB recovery trajectories](../figures/figure_recovery.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM4 | [Independent recovery error distributions](../figures/figure_ecdf.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM5 | [PB error distributions by dimension](../figures/figure_ecdf_pb_dimensions.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM6 | [Forcing budget and baseline comparison](../figures/figure_budget.pdf) | [build_budget.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_budget.py) |
| SM7 | [One-dimensional linear solution](../figures/figure_solution_1d.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM8 | [Higher-order PB recovery](../figures/figure_order.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM9 | [First-order PB solution](../figures/figure_solution_pb.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM10 | [Positive-error variance family](../figures/figure_variance_family.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |
| SM11 | [PB trajectories by ambient dimension](../figures/figure_dimension.pdf) | [build_supplement.py](../reproduction/workspace/paper_1111/revision-validation/publication-figures/build_supplement.py) |

## Reading and reproducing the records

The machine-readable [catalog](catalog.json) supplies the same experiment IDs, paper references and source paths. Paths in that file are relative to `reproduction/workspace/`. A **saved-record analysis** reads existing runs and rebuilds summaries; a **numerical validation** re-evaluates errors, derivatives, references or checks; a **fresh campaign** runs optimizers and may be expensive. Campaign scripts keep their original options and environment requirements: inspect their help, protocol and adjacent README before starting them. There is no single command here that silently reruns every campaign.

Supplementary Tables SM6–SM8 specify method settings, acceptance rules and benchmark protocols; they do not represent additional experiment cohorts. The outcome tables are mapped to the studies below.

The dated `summarize_phase1.py` summary intentionally includes historical averaged-evaluator rows. Current PB results come from the enlarged cohort with exact NC/Cd evaluators; the current full-span and 2D comparisons come from `exact_evaluator_summary.py`. The forcing-budget endpoint audit supersedes the original uncorrected budget analysis.

## Complete study index

| Study | Paper or supplement |
| --- | --- |
| [Independent second- and third-order recovery](#independent-recovery) | Sections 5.4, SM5.2 |
| [First-order Poisson–Boltzmann recovery in dimensions 5, 10 and 20](#pb-recovery) | Sections 5.1, 5.4, SM5.1.3, SM5.2 |
| [Reversible pair method and three comparators on the PB cohort](#reversible-pb) | Sections 5.3, SM2.4 |
| [Prescribed forcing with varying Gaussian budget](#forcing-budget) | Sections 5.5, 5.6, SM5.4 |
| [Pair tolerance, collapse and boundary-identification tests](#local-scaling) | Sections 5.2, SM5.1.1 |
| [Local objective, stopping and positive-floor controls](#objective-floors) | Sections 5.2, SM4.3 |
| [High-precision reference pair checks](#reversible-reference) | Sections SM2.2 |
| [Two rotating pairs for a nonlinear quadrature energy](#reversible-multipair) | Sections SM2.3 |
| [One-dimensional linear recovery with mixed collision orders](#linear-recovery) | Sections SM5.1.3, SM5.3.1, SM6.2 |
| [Target-informed one-dimensional second-order PB recovery](#second-order-recovery) | Sections SM5.1.3, SM5.3.1 |
| [Target-informed PB recovery through order three in dimension five](#higher-order-recovery) | Sections SM5.1.3, SM5.3.1, SM6.2 |
| [Full-span linear recovery in dimensions 10 and 20](#fullspan-recovery) | Sections SM5.1.3, SM5.3.1, SM5.3.2, SM6.2 |
| [Exact-parity, averaged and Taylor-series pair evaluator controls](#pair-evaluators) | Sections SM5.3.2 |
| [Positive-error Gaussian-variance family](#variance-family) | Sections 5.6, SM5.1.3, SM5.3.6 |
| [Two-dimensional recovery, mixed scales and prescribed forcing](#twod-controls) | Sections SM5.3.2, SM5.3.3 |
| [One-dimensional single-source forcing family](#forcing-single-source) | Sections SM5.1.2, SM5.3.4 |
| [One-dimensional multiple-source forcing controls](#forcing-multiple-sources) | Sections SM5.3.4 |
| [Prescribed Hermite–Gaussian multipole controls](#forcing-higher-order) | Sections SM5.3.4 |
| [Two-dimensional higher-order negative pilots](#twod-higher-order-negative) | Sections SM5.3.5 |
| [Completion, residual and frozen-chart endpoint validation](#endpoint-validation) | Sections SM6.1, SM6.2 |
| [Earlier serial timing measurements for NC and discrete completion](#controlled-timing) | Sections SM6.5 |
| [Pair constants and nonsmoothable cubic verification](#analytic-checks) | Sections SM1.2, SM1.5, SM2.2 |

## Main-paper studies

<a id="independent-recovery"></a>
### Independent second- and third-order recovery

Sections 5.4, SM5.2; Tables 1, SM13, SM14, SM15, SM17; Figures SM2, SM4.

100 matched starts for each derivative order, four methods. Retain all failed basins and both protocol versions: the supplement explicitly reports the earlier trial-failure rule and its 57 changed runs. The original ten-start guarded cohort is retained as reproduction evidence.

Saved evidence:

- [work/sinum-phase3-2026-11-11/cohort1d/results_amend2](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort1d/results_amend2)
- [work/sinum-phase3-2026-11-11/cohort1d/results](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort1d/results)
- [work/sinum-phase3-2026-11-11/prereg](../reproduction/workspace/work/sinum-phase3-2026-11-11/prereg)
- [work/sinum-phase3-2026-11-11/stats](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats)

Analysis and checks:

- Saved-record analysis: [analyze.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats/analyze.py).
- Saved-record analysis: [make_tables.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats/make_tables.py).
- Saved-record analysis: [main_recovery_table.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats/main_recovery_table.py). Rebuilds main Table 1 directly from all 700 relevant NC/C endpoints with coverage checks and input provenance.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_amend2.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort1d/run_amend2.py).

<a id="pb-recovery"></a>
### First-order Poisson–Boltzmann recovery in dimensions 5, 10 and 20

Sections 5.1, 5.4, SM5.1.3, SM5.2; Tables 1, SM12, SM16, SM17, SM20; Figures 1, SM2, SM3, SM5, SM9, SM11.

50 matched starts per dimension, four methods. The exact evaluator is primary; averaged-evaluator NC and discrete-completion arms are also explicitly reported controls. Figure 1 re-evaluates one saved d=10 endpoint pair; it is not another optimization study.

Saved evidence:

- [work/sinum-phase3-2026-11-11/cohort_pb/results](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/results)
- [work/sinum-phase3-2026-11-11/cohort_pb/results_nc_exact](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/results_nc_exact)
- [work/sinum-phase3-2026-11-11/cohort_pb/results_cd_exact](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/results_cd_exact)
- [numerical-section-revision/sinum-revision-2026-11-10/squared-gap/results](../reproduction/workspace/numerical-section-revision/sinum-revision-2026-11-10/squared-gap/results)

Analysis and checks:

- Saved-record analysis: [pb_table_rows.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats/pb_table_rows.py).
- Saved-record analysis: [main_recovery_table.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/stats/main_recovery_table.py). Rebuilds main Table 1 directly from all 700 relevant NC/C endpoints with coverage checks and input provenance.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [campaign.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/campaign.py).
- [campaign_nc_exact.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/campaign_nc_exact.py).
- [campaign_cd_exact.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/cohort_pb/campaign_cd_exact.py).

<a id="reversible-pb"></a>
### Reversible pair method and three comparators on the PB cohort

Sections 5.3, SM2.4; Tables SM3, SM4, SM5; Figures 2.

30 projected t runs, 18 projected h runs, and 150 runs in each L-BFGS-B chart. Figure 2 uses the 18 shared projected starts; cap failures stay in the denominator. Seed-9 tuning records document the stated exploratory selection.

Saved evidence:

- [work/alg41-pb-cohort-2026-11-12/results](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/results)
- [work/alg41-pb-cohort-2026-11-12/protocol.json](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/protocol.json)
- [work/alg41-pb-cohort-2026-11-12/hessian_check.json](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/hessian_check.json)
- [paper_1111/revision-validation/pair-coefficient-audit](../reproduction/workspace/paper_1111/revision-validation/pair-coefficient-audit)
- [work/alg41-pb-cohort-2026-11-12/pilot](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/pilot)

Analysis and checks:

- Saved-record analysis: [analyze.py](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/analyze.py).
- Numerical validation: [hessian_check.py](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/hessian_check.py). Finite-difference endpoint Hessians; does not rerun optimization.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [campaign.py](../reproduction/workspace/work/alg41-pb-cohort-2026-11-12/campaign.py).

<a id="forcing-budget"></a>
### Prescribed forcing with varying Gaussian budget

Sections 5.5, 5.6, SM5.4; Tables SM24; Figures 3, SM6.

Two forcings, P=2,4,6,8,12,16, ten starts and S/NC/C: all 360 optimizer endpoints plus fixed-center and P1 baselines. Retain every high-budget breakdown. P>=8 is exploratory endpoint evidence, excluded from optimizer-performance claims. The original analyze.py predates the corrected endpoint audit and is not the authoritative current summary.

Saved evidence:

- [work/pstudy-forcing-2026-11-12/results/runs](../reproduction/workspace/work/pstudy-forcing-2026-11-12/results/runs)
- [work/pstudy-forcing-2026-11-12/results/baselines](../reproduction/workspace/work/pstudy-forcing-2026-11-12/results/baselines)
- [work/pstudy-forcing-2026-11-12/protocol.json](../reproduction/workspace/work/pstudy-forcing-2026-11-12/protocol.json)
- [paper_1111/revision-validation/forcing-endpoint-audit](../reproduction/workspace/paper_1111/revision-validation/forcing-endpoint-audit)

Analysis and checks:

- Numerical validation: [analyze_endpoints.py](../reproduction/workspace/paper_1111/revision-validation/forcing-endpoint-audit/analyze_endpoints.py). Re-evaluates saved endpoints and independent references; produces the corrected energy-selection summary. Use the publication builder for the current figure.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [campaign.py](../reproduction/workspace/work/pstudy-forcing-2026-11-12/campaign.py).

## Local and reversible checks

<a id="local-scaling"></a>
### Pair tolerance, collapse and boundary-identification tests

Sections 5.2, SM5.1.1; Figures SM1.

Both prespecified joint optimization and exploratory coefficient-refitted sweeps are reported, including the one-start PB sweep. The nine successful local starts used for slopes do not replace the complete ten-start records.

Saved evidence:

- [work/sinum-phase3-2026-11-11/predictions/results](../reproduction/workspace/work/sinum-phase3-2026-11-11/predictions/results)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_toy.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/predictions/run_toy.py).
- [run_toy_vp.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/predictions/run_toy_vp.py).
- [run_pb_sweep.py](../reproduction/workspace/work/sinum-phase3-2026-11-11/predictions/run_pb_sweep.py).

<a id="objective-floors"></a>
### Local objective, stopping and positive-floor controls

Sections 5.2, SM4.3; Tables SM9, SM10, SM11.

Three local linear targets, ten matched starts, shifted-Gram versus oracle residual objectives and five squared-gap floors. Keep the disclosed one-start pilot and serial timing records.

Saved evidence:

- [paper_1111/revision-validation/objective-controls/results.json](../reproduction/workspace/paper_1111/revision-validation/objective-controls/results.json)
- [paper_1111/revision-validation/objective-controls/validation.json](../reproduction/workspace/paper_1111/revision-validation/objective-controls/validation.json)
- [paper_1111/revision-validation/objective-controls/timing-context.json](../reproduction/workspace/paper_1111/revision-validation/objective-controls/timing-context.json)
- [paper_1111/revision-validation/objective-controls/pilot](../reproduction/workspace/paper_1111/revision-validation/objective-controls/pilot)

Analysis and checks:

- Saved-record analysis: [build_text.py](../reproduction/workspace/paper_1111/revision-validation/objective-controls/build_text.py).
- Numerical validation: [validate_controls.py](../reproduction/workspace/paper_1111/revision-validation/objective-controls/validate_controls.py). 80-digit endpoint and refined-grid checks.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_controls.py](../reproduction/workspace/paper_1111/revision-validation/objective-controls/run_controls.py).

<a id="reversible-reference"></a>
### High-precision reference pair checks

Sections SM2.2; Tables SM1.

Five 70-digit pair iterations test closing, opening, both storage conversions and accepted decrease. Fresh execution performs optimization; the saved trajectories are immediately readable.

Saved evidence:

- [paper_1111/revision-validation/reversible_pair_checks.json](../reproduction/workspace/paper_1111/revision-validation/reversible_pair_checks.json)
- [paper_1111/revision-validation/reversible_pair_summary.csv](../reproduction/workspace/paper_1111/revision-validation/reversible_pair_summary.csv)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [verify_reversible_pairs.py](../reproduction/workspace/paper_1111/revision-validation/verify_reversible_pairs.py).

<a id="reversible-multipair"></a>
### Two rotating pairs for a nonlinear quadrature energy

Sections SM2.3; Tables SM2.

Three target-informed starts, reversible t and stable h: all six trajectories and serial timings. This is a discrete manufactured quadrature objective.

Saved evidence:

- [paper_1111/revision-validation/reversible-multipair/results.json](../reproduction/workspace/paper_1111/revision-validation/reversible-multipair/results.json)
- [paper_1111/revision-validation/reversible-multipair/results.csv](../reproduction/workspace/paper_1111/revision-validation/reversible-multipair/results.csv)
- [paper_1111/revision-validation/reversible-multipair/validation.json](../reproduction/workspace/paper_1111/revision-validation/reversible-multipair/validation.json)

Analysis and checks:

- Numerical validation: [verify.py](../reproduction/workspace/paper_1111/revision-validation/reversible-multipair/verify.py). 80-digit endpoint errors and quadrature-refit checks.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run.py](../reproduction/workspace/paper_1111/revision-validation/reversible-multipair/run.py).

## Additional recovery and evaluator controls

<a id="linear-recovery"></a>
### One-dimensional linear recovery with mixed collision orders

Sections SM5.1.3, SM5.3.1, SM6.2; Tables SM12, SM19, SM20; Figures SM3, SM7.

Ten shared starts, S/NC/C/Cd, one singleton plus first- and second-order collisions.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/linear1d](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/linear1d)

Analysis and checks:

- Saved-record analysis: [summarize_phase1.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/summarize_phase1.py). Summarizes several near-target families. Its PB and full-span rows are historical averaged-evaluator summaries; use the enlarged-cohort and exact-evaluator summaries for the current paper.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [chain_experiments.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/src/chain_experiments.py). Arguments: `--problem linear1d --seeds 0 10 --methods standard_restart stable adaptive tpair --jobs 1 --output fresh-results/linear1d`. Entire ten-start linear campaign; use a new output directory.

<a id="second-order-recovery"></a>
### Target-informed one-dimensional second-order PB recovery

Sections SM5.1.3, SM5.3.1; Tables SM12, SM19, SM20.

Ten shared starts. Current S uses the two-return policy in s_reruns; NC/Cd are guarded records and C uses the squared-gap chain solver.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/second1d](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/second1d)
- [numerical-section-revision/guarded-rerun-2026-11-09/directional/accuracy-second1d](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/directional/accuracy-second1d)
- [work/sinum-phase3-2026-11-11/s_reruns/results.json](../reproduction/workspace/work/sinum-phase3-2026-11-11/s_reruns/results.json)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [chain_experiments.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/src/chain_experiments.py). Arguments: `--problem second1d --seeds 40 50 --methods tpair --jobs 1 --output fresh-results/second1d`. C arm of the ten-start campaign; guarded sources provide the frozen NC/Cd arms.

<a id="higher-order-recovery"></a>
### Target-informed PB recovery through order three in dimension five

Sections SM5.1.3, SM5.3.1, SM6.2; Tables SM12, SM19, SM20; Figures SM8.

Ten shared starts, P=16, mixed second- and third-order target blocks. Preserve failed basins, rejected completions and historical S baselines.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/higher](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/higher)
- [numerical-section-revision/guarded-rerun-2026-11-09/directional/accuracy-higher](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/directional/accuracy-higher)
- [numerical-section-revision/guarded-rerun-2026-11-09/historical-baselines](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/historical-baselines)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [chain_experiments.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/src/chain_experiments.py). Arguments: `--problem higher --dimensions 5 --geometries 0 1 --seeds 10 15 --methods tpair --jobs 1 --output fresh-results/higher`. C arm on both geometries and five seeds; guarded sources provide NC/Cd.

<a id="fullspan-recovery"></a>
### Full-span linear recovery in dimensions 10 and 20

Sections SM5.1.3, SM5.3.1, SM5.3.2, SM6.2; Tables SM12, SM19, SM20, SM21, SM27.

Three starts per dimension, d pairs and two singletons. Exact and averaged NC/Cd evaluator records are both reported.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/fullspan](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/fullspan)
- [numerical-section-revision/phase1-2026-11-11/runs/evaluator-linear](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator-linear)
- [numerical-section-revision/guarded-rerun-2026-11-09/first-order/accuracy](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/first-order/accuracy)
- [numerical-section-revision/guarded-rerun-2026-11-09/historical-baselines](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/historical-baselines)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [first_order_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/first_order_runner.py). Arguments: `--mode linear_T --dimension 10 --geometry 0 --seed 10 --output fresh-results/span-d10-s10.json`. One-start example. Repeat for d=10,20 and seeds 10,11,12.
- [exact_evaluator_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/exact_evaluator_runner.py). Arguments: `linear --method stable --dimension 10 --geometry 0 --seed 10 --output fresh-results/span-nc-d10-s10.json`. One-start exact NC example. Repeat with method adaptive for Cd and across the same dimensions and seeds.

<a id="pair-evaluators"></a>
### Exact-parity, averaged and Taylor-series pair evaluator controls

Sections SM5.3.2; Tables SM21.

Matched evaluator reruns cover PB, full span, 2D controls and 20 one-dimensional directional starts. These are reported controls, not duplicate clutter.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/evaluator](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator)
- [numerical-section-revision/phase1-2026-11-11/runs/evaluator-chain](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator-chain)
- [numerical-section-revision/phase1-2026-11-11/runs/evaluator-linear](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator-linear)
- [numerical-section-revision/phase1-2026-11-11/runs/evaluator-twod](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator-twod)

Analysis and checks:

- Saved-record analysis: [exact_evaluator_summary.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/exact_evaluator_summary.py).
- Saved-record analysis: [evaluator_matched.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/evaluator_matched.py).

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [exact_evaluator_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/exact_evaluator_runner.py). Arguments: `linear --method stable --dimension 10 --geometry 0 --seed 10 --output fresh-results/evaluator-span.json`. One full-span evaluator example; twod --list enumerates two-dimensional tasks.
- [chain_series_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/chain_series_runner.py). Arguments: `--problem linear1d --seed 0 --output fresh-results/evaluator-chain.json`. One directional-series example. Use linear1d seeds 0–9 and second1d seeds 40–49 for the complete study.

<a id="variance-family"></a>
### Positive-error Gaussian-variance family

Sections 5.6, SM5.1.3, SM5.3.6; Tables SM23; Figures SM10.

Variance parameters 0.99,0.995,0.999, ten starts, four methods. Joint coefficient/geometry optimization and all unsuccessful basins are retained. The separate old normal-family experiment is not reported.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/variance](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/variance)
- [numerical-section-revision/guarded-rerun-2026-11-09/variance/results](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/variance/results)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [variance_t.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/variance_t.py).
- [run_matched.py](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/variance/run_matched.py).

## Prescribed forcing and negative controls

<a id="twod-controls"></a>
### Two-dimensional recovery, mixed scales and prescribed forcing

Sections SM5.3.2, SM5.3.3; Tables SM21, SM22.

Ten exact-recovery starts; ten starts at each scale mismatch (0.01,0.05); 9 smooth-source, 6 multiple-source and 3 exploratory forcing starts. Keep the explicitly reported tighter-allowance sensitivity control and fixed reference caches.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/runs/twod](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/twod)
- [numerical-section-revision/phase1-2026-11-11/runs/evaluator-twod](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/runs/evaluator-twod)
- [numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/accuracy](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/accuracy)
- [numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/tasks.json](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/tasks.json)
- [numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/historical/allowance](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/supplementary-2d/historical/allowance)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [twod_t_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/twod_t_runner.py). Arguments: `--id recovery_dipole_101_stable --output fresh-results/recovery-dipole-101-C.json`. One C example; --list enumerates all 48 task IDs.
- [exact_evaluator_runner.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/exact_evaluator_runner.py). Arguments: `twod --id recovery_dipole_101_stable --output fresh-results/recovery-dipole-101-NC.json`. One exact NC example; twod --list enumerates the NC/Cd task IDs.

<a id="forcing-single-source"></a>
### One-dimensional single-source forcing family

Sections SM5.1.2, SM5.3.4.

Ten independently parameterized P=2 cases; retain NC/Cd earlier records and the updated C endpoints. All have positive approximation error; completion does not improve it.

Saved evidence:

- [numerical-section-revision/sinum-revision-2026-11-10/forcing](../reproduction/workspace/numerical-section-revision/sinum-revision-2026-11-10/forcing)
- [work/independent-tests-2026-11-08/forcing_pair/holdout](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_pair/holdout)

Analysis and checks:

- Numerical validation: [revalidate.py](../reproduction/workspace/numerical-section-revision/sinum-revision-2026-11-10/forcing/revalidate.py). Recomputes endpoint and enlarged-domain reference checks.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_holdout.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_pair/run_holdout.py).

<a id="forcing-multiple-sources"></a>
### One-dimensional multiple-source forcing controls

Sections SM5.3.4.

Six families, 18 matched cases, 54 validated endpoints. Keep the nine original failed reference validations and their repaired-reference replacements; they are not 18 new scientific outcomes.

Saved evidence:

- [work/independent-tests-2026-11-08/forcing_1d/pilot](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_1d/pilot)
- [work/independent-tests-2026-11-08/forcing_1d/pilot_reference_repair](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_1d/pilot_reference_repair)
- [work/independent-tests-2026-11-08/forcing_1d/paired.csv](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_1d/paired.csv)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_screen.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_1d/run_screen.py).
- [rerun_reference.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_1d/rerun_reference.py).

<a id="forcing-higher-order"></a>
### Prescribed Hermite–Gaussian multipole controls

Sections SM5.3.4.

24 matched cases, 72 endpoints. All three exploratory settings are reported; preserve the original reference failures and resolved checks.

Saved evidence:

- [work/independent-tests-2026-11-08/forcing_chain/pilot](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/pilot)
- [work/independent-tests-2026-11-08/forcing_chain/pilot_grid_variant](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/pilot_grid_variant)
- [work/independent-tests-2026-11-08/forcing_chain/pilot_scale_variant](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/pilot_scale_variant)
- [work/independent-tests-2026-11-08/forcing_chain/paired.csv](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/paired.csv)

Analysis and checks:

- Numerical validation: [validate_resolved.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/validate_resolved.py). Validates saved optimizer endpoints against independent boundary-value references.

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run_chain.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/run_chain.py).
- [run_grid_variant.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/run_grid_variant.py).
- [run_scale_variant.py](../reproduction/workspace/work/independent-tests-2026-11-08/forcing_chain/run_scale_variant.py).

<a id="twod-higher-order-negative"></a>
### Two-dimensional higher-order negative pilots

Sections SM5.3.5.

36 runs: second/third derivatives, three seeds, reaction strengths 1 and 10, S/NC/Cd. The supplement explicitly reports failure to recover most targets. Unreported one-dimensional pilot1d and infrastructure smoke/failure directories are excluded.

Saved evidence:

- [work/independent-tests-2026-11-08/blind_higher_order/pilot](../reproduction/workspace/work/independent-tests-2026-11-08/blind_higher_order/pilot)
- [work/independent-tests-2026-11-08/blind_higher_order/pilot_strong](../reproduction/workspace/work/independent-tests-2026-11-08/blind_higher_order/pilot_strong)

Fresh campaign entry points (run from the script directory; retain the published environment and use fresh output paths):

- [run.py](../reproduction/workspace/work/independent-tests-2026-11-08/blind_higher_order/run.py).

## Validation and timing

<a id="endpoint-validation"></a>
### Completion, residual and frozen-chart endpoint validation

Sections SM6.1, SM6.2; Tables SM18, SM25, SM26, SM27.

These checks support existing experiment endpoints. They are not extra performance cohorts; preserve endpoint failures and numerical-noise classifications.

Saved evidence:

- [numerical-section-revision/phase1-2026-11-11/posthoc-stationarity.json](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/posthoc-stationarity.json)
- [numerical-section-revision/phase1-2026-11-11/pair-solver-residuals.json](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/pair-solver-residuals.json)
- [work/sinum-phase3-2026-11-11/stationarity](../reproduction/workspace/work/sinum-phase3-2026-11-11/stationarity)
- [numerical-section-revision/guarded-rerun-2026-11-09/independent/accuracy](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/independent/accuracy)

Analysis and checks:

- Numerical validation: [validate_phase1.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/validate_phase1.py). Solver regression, pair columns and slope tests.
- Numerical validation: [cd_exact_transfer_checks.py](../reproduction/workspace/numerical-section-revision/phase1-2026-11-11/cd_exact_transfer_checks.py). Rechecks accepted discrete transfers.

<a id="controlled-timing"></a>
### Earlier serial timing measurements for NC and discrete completion

Sections SM6.5; Tables SM28.

Three prescribed starts in each of nine problem/dimension groups. These timings concern NC and Cd, not the current adaptive C. Unexecuted phase3 timing campaign templates do not supply new reported timings.

Saved evidence:

- [numerical-section-revision/guarded-rerun-2026-11-09/first-order/benchmark](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/first-order/benchmark)
- [numerical-section-revision/guarded-rerun-2026-11-09/directional/timing-second1d](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/directional/timing-second1d)
- [numerical-section-revision/guarded-rerun-2026-11-09/directional/timing-higher](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/directional/timing-higher)
- [numerical-section-revision/guarded-rerun-2026-11-09/independent/benchmark](../reproduction/workspace/numerical-section-revision/guarded-rerun-2026-11-09/independent/benchmark)

<a id="analytic-checks"></a>
### Pair constants and nonsmoothable cubic verification

Sections SM1.2, SM1.5, SM2.2.

Supporting mathematical verification explicitly cited in the supplement; no separate solver-performance claim.

Saved evidence:

- [paper_1111/revision-validation/pair_analysis_checks.json](../reproduction/workspace/paper_1111/revision-validation/pair_analysis_checks.json)
- [paper_1111/revision-validation/nonsmoothable_cubic_summary.txt](../reproduction/workspace/paper_1111/revision-validation/nonsmoothable_cubic_summary.txt)

Analysis and checks:

- Numerical validation: [verify_pair_analysis.py](../reproduction/workspace/paper_1111/revision-validation/verify_pair_analysis.py). 80-digit analytic Gaussian correlations.
- Numerical validation: [verify_nonsmoothable_cubic.py](../reproduction/workspace/paper_1111/revision-validation/verify_nonsmoothable_cubic.py). Exact modular rank and symbolic polynomial checks.

## What is excluded

- October six-figure/six-table release, certificate experiment and coefficient-budget sweep not reported in the current manuscript.
- Old hierarchical-jet, normal-family and alternative-geometry campaigns not reported in the current manuscript.
- Unreported one-dimensional pilot1d runs, smoke tests, infrastructure launch failures and unused timing campaign outputs.
- Unused figure_reversible.pdf and editorial-figures replot; only the 14 figures actually included by the current sources are publication outputs.
- Draft manuscripts, referee notes, source-editing scripts, stale report PDFs, screenshots, caches and duplicate release trees.

Required solver dependencies remain available even when their earlier, unreported experiment records have been removed. Reported failures and unfavorable controls remain part of the scientific record.
