#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): the revised Algorithm 1 (pairs in t = h^2 >= 0) on the two-dimensional controls.

For every NC ("stable") task of guarded-rerun-2026-11-09/supplementary-2d/tasks.json in the cohorts
recovery, mismatch001, mismatch005, pilot-smoke, pilot-smooth and pilot-multi6, run the same case, seed,
order, perturbation and fit limit with the t-chart: squared_gap.Geometry (t in [0, (0.6 sigma)^2]) and the exact
parity Gram/column evaluator for every block list that contains a t-pair.  The frozen workspace modules are
imported unchanged and patched at run time; no irreversible transfer is made (method 'stable' + t-chart).
Usage: python3.14 twod_t_runner.py --id <task id of the NC task> --output runs/twod/<id>_T.json
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
SUPP = HERE.parent / 'guarded-rerun-2026-11-09' / 'supplementary-2d'
SQUARED = HERE.parent / 'sinum-revision-2026-11-10' / 'squared-gap'
COHORTS = ('recovery', 'mismatch001', 'mismatch005', 'pilot-smoke', 'pilot-smooth', 'pilot-multi6')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_t(modules, base):
    """t-aware Gram/columns in every given namespace; t-chart geometry in the base solver module."""
    import squared_gap
    for m in modules:
        if hasattr(m, 'block_gram'):
            og = m.block_gram
            m.block_gram = (lambda og: lambda b, other=None, **kw: (
                squared_gap.block_gram(b, other, **kw) if any('t' in x for x in b + (other or []))
                else og(b, other, **kw)))(og)
        if hasattr(m, 'column_values'):
            oc = m.column_values
            m.column_values = (lambda oc: lambda b, X, **kw: (
                squared_gap.column_values(b, X, **kw) if any('t' in x for x in b) else oc(b, X, **kw)))(oc)
    base.Geometry = squared_gap.Geometry


def tasks():
    out = []
    for t in json.loads((SUPP / 'tasks.json').read_text()):
        if t['cohort'] in COHORTS and t['method'] == 'stable' and t['energy_factor'] is None:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.list:
        for t in tasks():
            print(t['id'])
        return
    if args.output.exists():
        raise SystemExit(f'Output already exists: {args.output}')
    task = next(t for t in tasks() if t['id'] == args.id)
    root = SUPP / 'workspace'
    sys.path.insert(0, str(SQUARED))
    sys.path.insert(0, str(SUPP))                                       # pair_guard.py, as for run_case.py
    basepath = root / 'adaptive-desingularization/final-experiments/pb-independent-2d'
    if task['snapshot']:
        module = load('forcing_control', root / 'work/independent-tests-2026-11-08/forcing_2d' /
                      (task['snapshot'].replace('-', '_') + '.py'))
        ind = module.ind
        install_t([module, ind, ind.base], ind.base)
        build = lambda: module.Solver(task['case'], task['seed'], 'stable', task['max_fits'], task['order'])
    else:
        ind = load('independent_control', basepath / 'run.py')
        install_t([ind, ind.base], ind.base)
        build = lambda: ind.IndependentSolver(task['case'], task['seed'], 'stable', task['max_fits'],
                                              task['order'], task['perturbation'], 4, 'lattice')
    import numpy as np
    import scipy
    import torch
    import mpmath as mp
    started = time.perf_counter()
    solver = build()
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    result = solver.run()
    elapsed = time.perf_counter() - started
    if task['snapshot']:
        x, u, ux, uy, meta = module.reference_cached(task['case'], module.CASES[task['case']], n=512, L=10.)
        X, Y = np.meshgrid(x, x, indexing='ij')
        pts = np.column_stack((X.ravel(), Y.ravel()))
        v, g = module.values_grad(result['blocks'], np.asarray(result['coefficients']), pts)
        err2 = float(np.sum((v - u.ravel()) ** 2 + (g[:, 0] - ux.ravel()) ** 2 + (g[:, 1] - uy.ravel()) ** 2) * (x[1] - x[0]) ** 2)
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
    pairs = [b for b in solver.blocks if b['kind'] == 'pair']
    result.update(phase1_method='C (pairs in t = h^2 >= 0)', supplementary_task=task,
                  zero_squared_gaps=sum(b.get('t') == 0. for b in pairs),
                  squared_gaps=[b.get('t', b.get('h', 0.) ** 2) for b in pairs],
                  independent_endpoint_check=endpoint,
                  timing=dict(end_to_end_seconds=elapsed, context='shared host accuracy run; not a timing benchmark'))
    paths = sorted((SUPP / 'workspace').rglob('*.py')) + [SQUARED / 'squared_gap.py', Path(__file__)]
    result['reproduction'] = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                                  scipy=scipy.__version__, torch=torch.__version__, mpmath=mp.__version__,
                                  source_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ind.base.plain(result), indent=2) + '\n')
    print(json.dumps(dict(id=task['id'], error=result['final_relative_error'], endpoint=endpoint['relative_H1'],
                          fits=result['fits'], zeros=result['zero_squared_gaps'], stop=result['stop'],
                          seconds=round(elapsed, 1))), flush=True)


if __name__ == '__main__':
    main()
