#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): evaluator control for the full-span linear family and the two-dimensional controls.

NC ('stable') and the discrete pair completion C_d ('adaptive') keep the frozen half-gap chart h, but every pair
Gram matrix and column is evaluated by the exact parity formula of squared_gap.py, the evaluator of C.  This is the
patch of first_order_runner.py --mode pb_NC_analytic / pb_Cd_analytic (which stays frozen), applied to the linear
first-order solver and to the two-dimensional solvers.  The frozen sources are imported unchanged and patched at
run time; cases, seeds, orders, perturbations, fit limits and stopping rules are those of the frozen tasks.
Usage:
  python3.14 exact_evaluator_runner.py linear --method stable --dimension 10 --seed 10 --output runs/evaluator-linear/x.json
  python3.14 exact_evaluator_runner.py twod --id recovery_dipole_101_adaptive --output runs/evaluator-twod/x.json
  python3.14 exact_evaluator_runner.py twod --list
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARDED = HERE.parent / 'guarded-rerun-2026-11-09' / 'first-order'
SUPP = HERE.parent / 'guarded-rerun-2026-11-09' / 'supplementary-2d'
SQUARED = HERE.parent / 'sinum-revision-2026-11-10' / 'squared-gap'
COHORTS = ('recovery', 'mismatch001', 'mismatch005', 'pilot-smoke', 'pilot-smooth', 'pilot-multi6')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_exact(modules):
    """Exact parity Gram/columns for every block list, in every given namespace; the h-chart is unchanged."""
    import squared_gap
    for m in modules:
        if hasattr(m, 'block_gram'):
            m.block_gram = lambda b, other=None, **kw: squared_gap.block_gram(b, other, **kw)
        if hasattr(m, 'column_values'):
            m.column_values = lambda b, X, **kw: squared_gap.column_values(b, X, **kw)


def twod_tasks():
    return [t for t in json.loads((SUPP / 'tasks.json').read_text())
            if t['cohort'] in COHORTS and t['energy_factor'] is None]


def run_linear(args):
    sys.path.insert(0, str(SQUARED))
    sys.path.insert(0, str(GUARDED))                     # pair_guard.py
    sys.path.insert(0, str(GUARDED / 'source' / 'final-experiments' / 'linear'))
    import run as module
    install_exact([module, sys.modules['base_solver']])
    task = dict(family='complexity', dimension=args.dimension, geometry=args.geometry, seed=args.seed,
                method=args.method)
    start = time.perf_counter()
    solver = module.Solver(task, 1200)
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    result = solver.run()
    elapsed = time.perf_counter() - start
    independent = module.high_precision_error(solver.blocks, solver.coef, solver.tb, solver.tc, dps=110)
    assert abs(independent['relative'] - result['final_relative_error']) <= max(1e-20, 1e-10 * independent['relative'])
    result['independent_endpoint_check_110_digits'] = independent
    paths = sorted((GUARDED / 'source').rglob('*.py')) + [GUARDED / 'pair_guard.py']
    tag = dict(family='linear full span', dimension=args.dimension, geometry=args.geometry, seed=args.seed)
    return module, module.plain, result, elapsed, paths, tag, solver


def run_twod(args):
    task = next(t for t in twod_tasks() if t['id'] == args.id)
    root = SUPP / 'workspace'
    sys.path.insert(0, str(SQUARED))
    sys.path.insert(0, str(SUPP))                        # pair_guard.py, as for run_case.py
    basepath = root / 'adaptive-desingularization/final-experiments/pb-independent-2d'
    if task['snapshot']:
        module = load('forcing_control', root / 'work/independent-tests-2026-11-08/forcing_2d' /
                      (task['snapshot'].replace('-', '_') + '.py'))
        ind = module.ind
        install_exact([module, ind, ind.base])
        build = lambda: module.Solver(task['case'], task['seed'], task['method'], task['max_fits'], task['order'])
    else:
        module = ind = load('independent_control', basepath / 'run.py')
        install_exact([ind, ind.base])
        build = lambda: ind.IndependentSolver(task['case'], task['seed'], task['method'], task['max_fits'],
                                              task['order'], task['perturbation'], 4, 'lattice')
    import numpy as np
    import mpmath as mp
    start = time.perf_counter()
    solver = build()
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    result = solver.run()
    elapsed = time.perf_counter() - start
    if task['snapshot']:
        x, u, ux, uy, meta = module.reference_cached(task['case'], module.CASES[task['case']], n=512, L=10.)
        X, Y = np.meshgrid(x, x, indexing='ij')
        pts = np.column_stack((X.ravel(), Y.ravel()))
        v, g = module.values_grad(result['blocks'], np.asarray(result['coefficients']), pts)
        err2 = float(np.sum((v - u.ravel()) ** 2 + (g[:, 0] - ux.ravel()) ** 2 + (g[:, 1] - uy.ravel()) ** 2)
                     * (x[1] - x[0]) ** 2)
        endpoint = dict(kind='direct refined Fourier grid', relative_H1=math.sqrt(err2 / meta['norm2']))
    else:
        from kernels import _mp_primitives
        with mp.workdps(110):
            A = [(*p, mp.mpf(.45)) for p in _mp_primitives(solver.blocks, solver.coef)]
            T = ind.target_primitives(solver.terms)
            norm2 = ind.inner(T, T)
            err2 = ind.inner(A, A) - 2 * ind.inner(A, T) + norm2
            endpoint = dict(kind='analytic mixed-width Gaussian H1, 110 decimal digits',
                            relative_H1=float(mp.sqrt(err2 / norm2)))
        assert abs(endpoint['relative_H1'] - result['final_relative_error']) <= max(1e-20, 1e-10 * endpoint['relative_H1'])
    result.update(supplementary_task=task, independent_endpoint_check=endpoint)
    paths = sorted((SUPP / 'workspace').rglob('*.py')) + [SUPP / 'pair_guard.py']
    return module, ind.base.plain, result, elapsed, paths, dict(family='two-dimensional', task=task['id']), solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family', choices=['linear', 'twod'])
    ap.add_argument('--method', choices=['stable', 'adaptive'])
    ap.add_argument('--dimension', type=int)
    ap.add_argument('--geometry', type=int, default=0)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--id')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.list:
        for t in twod_tasks():
            print(t['id'])
        return
    if args.output.exists():
        raise SystemExit(f'Output already exists: {args.output}')
    module, plain, result, elapsed, paths, tag, solver = (run_linear if args.family == 'linear' else run_twod)(args)
    import numpy as np
    import scipy
    import torch
    import mpmath
    method = result.get('method', args.method or tag.get('task', '').rsplit('_', 1)[-1])
    pairs = [b for b in solver.blocks if b['kind'] == 'pair']
    result.update(phase1_control='exact parity evaluator in the h-chart (evaluator of C)', phase1_tag=tag,
                  completed_pairs=sum(bool(b.get('completed')) for b in pairs), open_pairs=sum(not b.get('completed') for b in pairs),
                  timing=dict(end_to_end_seconds=elapsed, context='shared host accuracy run; not a timing benchmark'))
    paths = paths + [SQUARED / 'squared_gap.py', Path(__file__)]
    result['reproduction'] = dict(command=sys.argv, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, mpmath=mpmath.__version__,
        torch_threads=torch.get_num_threads(),
        source_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(result), indent=2) + '\n')
    print(json.dumps(dict(tag=tag, method=method, error=result['final_relative_error'], fits=result['fits'],
                          stop=result.get('stop'), seconds=round(elapsed, 2))), flush=True)


if __name__ == '__main__':
    main()
