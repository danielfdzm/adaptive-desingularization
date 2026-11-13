#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): post-run checks of every accepted pair completion of C_d with the exact pair evaluator.

The checks are those of the frozen guarded run_case.py, applied to the recorded events: for each completion, the
110-digit norm of the state change is at most the recorded state bound and the state tolerance; in the linear family
the 110-digit quadratic energy increment is also at most the allowance.  Records: full span
(runs/evaluator-linear/linear_adaptive_exact_*.json) or first-order PB (Phase 3, cohort_pb/results_cd_exact).
One family per process (both families name their modules run/kernels).
Usage: python3.14 cd_exact_transfer_checks.py --family linear|pb --output cd-exact-transfer-checks-<family>.json
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARDED = HERE.parent / 'guarded-rerun-2026-11-09' / 'first-order'
PB_RECORDS = HERE.parents[1] / 'work' / 'sinum-phase3-2026-11-11' / 'cohort_pb' / 'results_cd_exact'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['linear', 'pb'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(GUARDED))
    sys.path.insert(0, str(GUARDED / 'source' / 'final-experiments' / args.family))
    import mpmath
    import run as module
    from kernels import _mp_primitives, _mp_inner
    files = (sorted((HERE / 'runs' / 'evaluator-linear').glob('linear_adaptive_exact_*.json')) if args.family == 'linear'
             else sorted(p for p in PB_RECORDS.glob('pb_d*.json')))
    rows, n_events = [], 0
    for f in files:
        r = json.loads(f.read_text())
        if args.family == 'linear':
            tag = r['phase1_tag']
            task = dict(family='complexity', dimension=tag['dimension'], geometry=tag['geometry'], seed=tag['seed'],
                        method='adaptive')
            solver = module.Solver(task, 1200)
        else:
            assert r.get('phase1_mode') == 'pb_Cd_analytic' and r.get('method') == 'adaptive', f
            solver = module.Solver(r['dimension'], r['seed'], 'adaptive', 12, 1500, r['geometry_id'], 8128, 1.)
        checks = []
        for event in r['events']:
            if event['kind'] != 'completion':
                continue
            i = next(k for k, b in enumerate(event['before_blocks']) if b['id'] == event['block_id'])
            ix = module.offsets(event['before_blocks'])
            c = event['before_coefficients'][ix[i]:ix[i + 1]]
            before_block, trial_block = [event['before_blocks'][i]], [event['trial_blocks'][i]]
            actual = module.high_precision_error(trial_block, c, before_block, c, dps=110)['norm']
            bound_ok = actual <= event['state_error_bound'] * (1 + 1e-8) + 1e-14 * solver.target2 ** .5
            tol_ok = actual <= event['state_tolerance'] * (1 + 1e-8)
            with mpmath.workdps(110):
                negate = lambda p: [(cc, [-x for x in v], -a) for cc, v, a in p]
                delta = _mp_primitives(trial_block, c) + negate(_mp_primitives(before_block, c))
                residual = _mp_primitives(event['before_blocks'], event['before_coefficients']) + negate(
                    _mp_primitives(solver.tb, solver.tc))
                quadratic = float(_mp_inner(residual, delta, module.SIGMA) + _mp_inner(delta, delta, module.SIGMA) / 2)
            allowance_ok = (quadratic <= event['eps_j'] + 1e-12 * solver.target2) if args.family == 'linear' else None
            checks.append(dict(block_id=event['block_id'], fits=event['fits'], state_change_norm_110_digits=actual,
                               state_error_bound=event['state_error_bound'], state_tolerance=event['state_tolerance'],
                               state_bound_pass=bool(bound_ok), state_tolerance_pass=bool(tol_ok),
                               quadratic_energy_increment_110_digits=quadratic, allowance=event['eps_j'],
                               linear_allowance_pass=allowance_ok))
        n_events += len(checks)
        rows.append(dict(file=str(f.relative_to(HERE.parents[1])), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                         completions=len(checks), all_pass=all(x['state_bound_pass'] and x['state_tolerance_pass'] and
                                                               x['linear_allowance_pass'] is not False for x in checks),
                         checks=checks))
        print(f.name, len(checks), rows[-1]['all_pass'], flush=True)
    out = dict(family=args.family, records=len(rows), completions=n_events,
               all_pass=all(x['all_pass'] for x in rows), rows=rows)
    args.output.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
