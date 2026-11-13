#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): first-order pair-solver runs that need the squared-gap chart or its evaluator.

Modes (one case per process; the frozen guarded sources are imported unchanged and patched at run time):
  linear_T      full-span linear family: pairs in t = h^2 >= 0 (new C of the revised Algorithm 1), with the
                exact parity Gram of squared_gap.py; no irreversible transfer (method 'stable' + t-chart)
  pb_NC_analytic, pb_Cd_analytic
                first-order PB: NC and the discrete pair completion C_d in the h-chart, but with the same exact
                parity Gram/column evaluator as T (evaluator control for the first-order cohort)
Usage:
  python3.14 first_order_runner.py --mode linear_T --dimension 10 --seed 10 --output runs/fullspan/x.json
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARDED = HERE.parent / 'guarded-rerun-2026-11-09' / 'first-order'
SQUARED = HERE.parent / 'sinum-revision-2026-11-10' / 'squared-gap'


def install_linear_t(module):
    """t-chart for the linear first-order solver: same patch as squared_gap.install for PB, applied to the
    namespaces that the linear solver uses (run, base_solver)."""
    import squared_gap
    base = sys.modules['base_solver']
    original = module.block_gram

    def gram(b, other=None, **kw):
        if any('t' in x for x in b + (other or [])):
            return squared_gap.block_gram(b, other, **kw)
        return original(b, other, **kw)
    module.block_gram = gram
    base.block_gram = gram
    module.Geometry = squared_gap.Geometry
    base.Geometry = squared_gap.Geometry


def install_pb_analytic(module):
    """Exact parity evaluator for every pair (h-chart unchanged): E_t, D_t with t = h^2."""
    import squared_gap
    module.block_gram = lambda b, other=None, **kw: squared_gap.block_gram(b, other, **kw)
    module.column_values = lambda b, X, **kw: squared_gap.column_values(b, X, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['linear_T', 'pb_NC_analytic', 'pb_Cd_analytic'], required=True)
    ap.add_argument('--dimension', type=int, required=True)
    ap.add_argument('--geometry', type=int, default=0)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f'Output already exists: {args.output}')
    family = 'linear' if args.mode == 'linear_T' else 'pb'
    sys.path.insert(0, str(SQUARED))
    sys.path.insert(0, str(GUARDED))                     # pair_guard.py (as for the guarded run_case.py)
    sys.path.insert(0, str(GUARDED / 'source' / 'final-experiments' / family))
    import run as module
    import numpy as np
    import scipy
    import torch
    import mpmath
    from threadpoolctl import threadpool_info
    if args.mode == 'linear_T':
        install_linear_t(module)
        task = dict(family='complexity', dimension=args.dimension, geometry=args.geometry, seed=args.seed,
                    method='stable')
        build = lambda: module.Solver(task, 1200)
    else:
        install_pb_analytic(module)
        method = 'stable' if args.mode == 'pb_NC_analytic' else 'adaptive'
        build = lambda: module.Solver(args.dimension, args.seed, method, 12, 1500, args.geometry, 8128, 1.)
    start = time.perf_counter()
    solver = build()
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    result = solver.run()
    elapsed = time.perf_counter() - start
    independent = module.high_precision_error(solver.blocks, solver.coef, solver.tb, solver.tc, dps=110)
    assert abs(independent['relative'] - result['final_relative_error']) <= max(1e-20, 1e-10 * independent['relative'])
    pairs = [b for b in solver.blocks if b['kind'] == 'pair']
    result['phase1_mode'] = args.mode
    result['zero_squared_gaps'] = sum(b.get('t') == 0. for b in pairs)
    result['squared_gaps'] = [b.get('t', b.get('h', 0.) ** 2) for b in pairs]
    result['independent_endpoint_check_110_digits'] = independent
    result['timing'] = dict(end_to_end_seconds=elapsed, context='shared host, accuracy campaign; not a timing benchmark')
    paths = sorted((GUARDED / 'source').rglob('*.py')) + [SQUARED / 'squared_gap.py', Path(__file__)]
    result['reproduction'] = dict(command=sys.argv, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, mpmath=mpmath.__version__,
        threadpools=threadpool_info(), torch_threads=torch.get_num_threads(),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(module.plain(result), indent=2) + '\n')
    print(json.dumps(dict(mode=args.mode, d=args.dimension, g=args.geometry, seed=args.seed,
                          error=result['final_relative_error'], fits=result['fits'], stop=result['stop'],
                          zeros=result['zero_squared_gaps'], seconds=round(elapsed, 2))), flush=True)


if __name__ == '__main__':
    main()
