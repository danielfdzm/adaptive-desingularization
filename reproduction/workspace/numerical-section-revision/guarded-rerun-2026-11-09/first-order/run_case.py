#!/usr/bin/env python3
"""Run one unchanged numerical configuration with guarded pair formation."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['pb', 'linear'], required=True)
    ap.add_argument('--dimension', type=int, required=True)
    ap.add_argument('--geometry', type=int, default=0)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--method', choices=['standard', 'stable', 'adaptive'], required=True)
    ap.add_argument('--max-fits', type=int)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timing-context', default='development pilot; host not exclusive')
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f'Output already exists: {args.output}')
    sys.path.insert(0, str(HERE / 'source' / 'final-experiments' / args.family))
    import run as module
    import numpy as np
    import scipy
    import torch
    import mpmath
    from threadpoolctl import threadpool_info
    if args.family == 'pb':
        build = lambda: module.Solver(args.dimension, args.seed, args.method, 12,
                                      args.max_fits or 1500, args.geometry, 8128, 1.)
    else:
        task = dict(family='complexity', dimension=args.dimension,
                    geometry=args.geometry, seed=args.seed, method=args.method)
        build = lambda: module.Solver(task, args.max_fits or 1200)
    start = time.perf_counter()
    solver = build()
    setup = time.perf_counter() - start
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    original_record = solver.record
    record_seconds = 0.
    def record(*a, **kw):
        nonlocal record_seconds
        t = time.perf_counter()
        try:
            return original_record(*a, **kw)
        finally:
            record_seconds += time.perf_counter() - t
    solver.record = record
    result = solver.run()
    elapsed = time.perf_counter() - start
    # Independent endpoint accuracy check is deliberately outside solver timing.
    audit_start = time.perf_counter()
    independent = module.high_precision_error(solver.blocks, solver.coef,
                                               solver.tb, solver.tc, dps=110)
    transfer_checks = []
    from kernels import _mp_primitives, _mp_inner
    for event in result['events']:
        if event['kind'] != 'completion':
            continue
        i = next(k for k, b in enumerate(event['before_blocks']) if b['id'] == event['block_id'])
        ix = module.offsets(event['before_blocks'])
        c = event['before_coefficients'][ix[i]:ix[i+1]]
        before_block = [event['before_blocks'][i]]
        trial_block = [event['trial_blocks'][i]]
        actual = module.high_precision_error(trial_block, c, before_block, c, dps=110)['norm']
        assert actual <= event['state_error_bound'] * (1+1e-8) + 1e-14 * solver.target2**.5
        assert actual <= event['state_tolerance'] * (1+1e-8)
        with mpmath.workdps(110):
            negate = lambda p: [(c, [-x for x in v], -a) for c, v, a in p]
            delta = _mp_primitives(trial_block, c) + negate(_mp_primitives(before_block, c))
            residual = _mp_primitives(event['before_blocks'], event['before_coefficients']) + negate(
                _mp_primitives(solver.tb, solver.tc))
            quadratic_increment = float(_mp_inner(residual, delta, module.SIGMA) +
                                        _mp_inner(delta, delta, module.SIGMA)/2)
        if args.family == 'linear':
            assert quadratic_increment <= event['eps_j'] + 1e-12 * solver.target2
        transfer_checks.append(dict(block_id=event['block_id'], fits=event['fits'],
            state_change_norm_110_digits=actual, state_bound_pass=True,
            quadratic_energy_increment_110_digits=quadratic_increment,
            recorded_gate=event['gate'], allowance=event['eps_j']))
    audit_seconds = time.perf_counter() - audit_start
    assert abs(independent['relative'] - result['final_relative_error']) <= max(
        1e-20, 1e-10 * independent['relative'])
    result['guard_counts'] = solver.guard_counts
    result['directional_guard'] = dict(applicable=False,
        reason='These solvers contain only singles and pairs; no directional joins or higher-order blocks.')
    result['timing'] = dict(context=args.timing_context, setup_seconds=setup,
        end_to_end_seconds=elapsed, diagnostic_record_seconds=record_seconds,
        end_to_end_without_diagnostic_records_seconds=elapsed-record_seconds,
        independent_endpoint_audit_seconds=audit_seconds,
        definition='perf_counter from Solver construction through run return; includes setup, scans, transfer tests, refits and in-run diagnostics; excludes process/import overhead and independent post-run 110-digit endpoint audit.')
    result['independent_endpoint_check_110_digits'] = independent
    result['independent_transfer_checks'] = transfer_checks
    paths = list((HERE / 'source').rglob('*.py')) + [HERE/'pair_guard.py', Path(__file__)]
    result['reproduction'] = dict(command=sys.argv, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
        mpmath=mpmath.__version__, threadpools=threadpool_info(),
        torch_threads=torch.get_num_threads(), pid=os.getpid(),
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(module.plain(result), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), family=args.family, dimension=args.dimension,
                          method=args.method, error=result['final_relative_error'], fits=result['fits'],
                          seconds=elapsed, diagnostics_seconds=record_seconds,
                          guard_counts=solver.guard_counts)), flush=True)


if __name__ == '__main__':
    main()
