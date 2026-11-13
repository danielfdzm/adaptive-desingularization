#!/usr/bin/env python3
"""Final stationarity residual of every Phase-3 run (plan step 7).

Cohort A (1D independent recovery; S, NC, C_d, C): the recorded final state is rebuilt in the Phase-1
chain solver with the cohort's problem (sigma 0.65, lambda 10, 384-node Gauss-Legendre rule on [-10,10],
centre box 3) and the record's method, and safeguard_cleanup.cleanup(solver, tol=1e-7, max_evals=200)
is called. R0 is its entry residual (projected gradient, gamma = 1, unit scale); the cleanup stop says
whether the endpoint is stationary to tol in the frozen chart. Recorded endpoints are not changed.
Cohort B (first-order PB; S, NC, C_d, C): stored fields only (projected_gradient_max_abs of the accepted
state, coefficient-fit KKT residual, stop reason).
Writes stationarity-rows.json and stationarity-rows.csv.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
P1 = ROOT / 'numerical-section-revision' / 'phase1-2026-11-11' / 'src'
sys.path.insert(0, str(P1))
import chain_solver as cs              # noqa: E402
import safeguard_cleanup as sc         # noqa: E402
from scipy.special import roots_legendre  # noqa: E402

LABEL_1D = dict(S='S', NC='NC', Cd='C_d', C='C')      # amendment 2: cohort1d/results_amend2 (paper labels)
LABEL_PB = dict(S='S', NC='NC_averaged', C='C_d_averaged', T='C')     # frozen evaluator (amendment 3: secondary)
EXTRA_PB = (('results_nc_exact', 'NCx', 'NC'), ('results_cd_exact', 'Cdx', 'C_d'))   # exact pair evaluator


def case(order, seed):
    sigma = .65
    rng = np.random.default_rng(100000 + seed)
    shift, factor = float(rng.uniform(-.6, .6)), float(rng.uniform(.7, 1.3))
    target = (np.array([[shift]]), np.array([[1.]]), np.array([order]), np.array([factor * sigma ** order]))
    starts = np.random.default_rng(seed).uniform(-1.2, 1.2, (order + 1, 1))
    x, w = roots_legendre(384)
    return sigma, target, starts, 10. * x[:, None], 10. * w


def rebuild(rec):
    sigma, target, starts, X, W = case(rec['order'], rec['seed'])
    s = cs.Solver(1, sigma, target, starts, rec['method'], X, W, lam=10., max_fits=10 ** 9, box=3.)
    blocks = []
    for b in rec['final_blocks']:
        nb = {k: (np.asarray(v, dtype=float) if k in ('c', 'e', 's', 'p') else v) for k, v in b.items()}
        if 'p' in nb:
            nb['p'] = nb['p'].reshape(-1, 1)
        blocks.append(nb)
    s.blocks = blocks
    s.coef = np.asarray(rec['final_coefficients'], dtype=float)
    s.warm = s.coef.copy()
    return s


def main():
    rows = []
    reuse = '--pb-only' in sys.argv and (HERE / 'stationarity-rows.json').exists()
    if reuse:
        rows = [r for r in json.loads((HERE / 'stationarity-rows.json').read_text())['rows'] if r['cohort'] == '1D independent']
    for order in ((2, 3) if not reuse else ()):
        for seed in range(900, 1000):
            for suffix, label in LABEL_1D.items():
                f = BASE / 'cohort1d' / 'results_amend2' / f'order{order}_seed{seed}_{suffix}.json'
                rec = json.loads(f.read_text())
                row = dict(cohort='1D independent', order=order, seed=seed, method=label, record=str(f.relative_to(ROOT)),
                           sha256=hashlib.sha256(f.read_bytes()).hexdigest(), stop_main=rec['stop'],
                           error=rec['final_relative_error'])
                try:
                    s = rebuild(rec)
                    s.fits = 0
                    s.max_fits = 200
                    out = sc.cleanup(s, tol=1e-7, max_evals=200)
                    row.update(R0=out.get('initial', {}).get('pg_max'), cleanup_stop=out.get('stop'),
                               R_final=out.get('final', {}).get('pg_max'), cleanup_iterations=out.get('iterations'),
                               cleanup_evaluations=out.get('evaluations'), objective_noise=out.get('objective_noise'),
                               trial_fit_failures=out.get('trial_fit_failures'),
                               noise_probe_failed=out.get('noise_probe_failed'),
                               main_failed_trials=sum(v for k, v in (rec.get('guard_counts') or {}).items()
                                                      if str(k).startswith('optimizer_trial:')))
                except Exception as exc:                       # keep the row, record why the rebuild failed
                    row.update(R0=None, cleanup_stop=f'rebuild failed: {exc!r}', R_final=None)
                rows.append(row)
                print(f"1D r={order} seed={seed} {label}: R0={row['R0']} stop={row['cleanup_stop']}", flush=True)
    for d in (5, 10, 20):
        for g in (0, 1):
            for seed in range(10, 35):
                for sub, suffix, label in [('results', k, v) for k, v in LABEL_PB.items()] + list(EXTRA_PB):
                    f = BASE / 'cohort_pb' / sub / f'pb_d{d}_g{g}_s{seed}_{suffix}.json'
                    if not f.exists():
                        continue
                    rec = json.loads(f.read_text())
                    fit = rec.get('coefficient_fit') or {}
                    rows.append(dict(cohort='PB first order', dimension=d, geometry=g, seed=seed, method=label,
                                     record=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                     stop_main=rec['stop'], error=rec['final_relative_error'],
                                     R0=rec.get('projected_gradient_max_abs'),
                                     coefficient_kkt=fit.get('kkt_residual') if isinstance(fit, dict) else None,
                                     cleanup_stop='not run (stored fields)', R_final=None))
    meta = dict(cleanup_sha256=hashlib.sha256((P1 / 'safeguard_cleanup.py').read_bytes()).hexdigest(),
                solver_sha256=hashlib.sha256((P1 / 'chain_solver.py').read_bytes()).hexdigest(),
                tol=1e-7, max_evals=200, rows=len(rows))
    (HERE / 'stationarity-rows.json').write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=float) + '\n')
    keys = ['cohort', 'order', 'dimension', 'geometry', 'seed', 'method', 'stop_main', 'error', 'R0', 'coefficient_kkt',
            'cleanup_stop', 'R_final', 'cleanup_iterations', 'cleanup_evaluations', 'trial_fit_failures',
            'noise_probe_failed', 'main_failed_trials', 'record', 'sha256']
    with (HERE / 'stationarity-rows.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
