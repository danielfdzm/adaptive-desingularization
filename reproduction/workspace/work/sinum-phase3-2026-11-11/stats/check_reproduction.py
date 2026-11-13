#!/usr/bin/env python3
"""Pre-registered reproduction checks: the reruns of existing starts must match the stored records.

Cohort A, seeds 900-909: NC/C against the guarded confirmation records, S against the historical
S records. Cohort B, seeds 10-14: NC/C against the guarded first-order records, T against the
squared-gap records, and S (two-return rule) against the historical first-return S records,
which can differ only where S returned locally before the fit limit.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
NSR = ROOT / 'numerical-section-revision'


def strip(events):
    return [{k: v for k, v in e.items() if k not in ('wall', 'elapsed_seconds')} for e in events]


def main():
    out = dict(cohort1d=[], cohort_pb=[])
    keys1 = ('final_relative_error', 'fits', 'iterations', 'stop', 'final_coefficients', 'final_blocks',
             'completions', 'joins')
    for order in (2, 3):
        for seed in range(900, 910):
            for m, ref in (('S', ROOT / 'work/independent-tests-2026-11-08/blind_higher_order/holdout' /
                            f'order{order}_seed{seed}_standard_restart.json'),
                           ('NC', NSR / 'guarded-rerun-2026-11-09/independent/accuracy' / f'order{order}_seed{seed}_stable.json'),
                           ('C', NSR / 'guarded-rerun-2026-11-09/independent/accuracy' / f'order{order}_seed{seed}_adaptive.json')):
                a = json.loads((BASE / 'cohort1d/results' / f'order{order}_seed{seed}_{m}.json').read_text())
                b = json.loads(ref.read_text())
                eq = {k: a[k] == b[k] for k in keys1}
                out['cohort1d'].append(dict(order=order, seed=seed, method=m, identical=all(eq.values()), fields=eq))
    keysb = ('final_relative_error', 'fits', 'iterations', 'stop', 'coefficients', 'blocks', 'completions')
    for d in (5, 10, 20):
        for g in (0, 1):
            for seed in range(10, 15):
                refs = dict(NC=NSR / 'guarded-rerun-2026-11-09/first-order/accuracy' / f'pb_d{d}_g{g}_s{seed}_stable.json',
                            C=NSR / 'guarded-rerun-2026-11-09/first-order/accuracy' / f'pb_d{d}_g{g}_s{seed}_adaptive.json',
                            T=NSR / 'sinum-revision-2026-11-10/squared-gap/results' / f'pb_d{d}_g{g}_s{seed}_squared_gap.json',
                            S=NSR / 'guarded-rerun-2026-11-09/historical-baselines' / f'pb_d{d}_g{g}_s{seed}_standard.json')
                for m, ref in refs.items():
                    pa = BASE / 'cohort_pb/results' / f'pb_d{d}_g{g}_s{seed}_{m}.json'
                    if not pa.exists():
                        continue
                    a = json.loads(pa.read_text())
                    b = json.loads(ref.read_text())
                    eq = {k: a[k] == b[k] for k in keysb}
                    eq['events'] = strip(a['events']) == strip(b['events'])
                    row = dict(dimension=d, geometry=g, seed=seed, method=m, identical=all(eq.values()), fields=eq,
                               new_error=a['final_relative_error'], new_fits=a['fits'], new_stop=a['stop'],
                               old_error=b['final_relative_error'], old_fits=b['fits'], old_stop=b['stop'])
                    row['expected_identical'] = (b['stop'] == 'work_limit') if m == 'S' else True
                    out['cohort_pb'].append(row)
    # Amendment 2: runs without a failed trial must reproduce their pre-amendment records exactly.
    out['amend2'] = []
    pre = dict(S='S', NC='NC', Cd='C', C='tpair')
    for order in (2, 3):
        for seed in range(900, 1000):
            for m, old in pre.items():
                pa = BASE / 'cohort1d/results_amend2' / f'order{order}_seed{seed}_{m}.json'
                if not pa.exists():
                    continue
                a = json.loads(pa.read_text())
                b = json.loads((BASE / 'cohort1d/results' / f'order{order}_seed{seed}_{old}.json').read_text())
                failed = sum(v for k, v in (a.get('guard_counts') or {}).items() if str(k).startswith('optimizer_trial:'))
                eq = {k: a[k] == b[k] for k in keys1}
                out['amend2'].append(dict(order=order, seed=seed, method=m, failed_trials=failed, identical=all(eq.values()),
                                          old_error=b['final_relative_error'], new_error=a['final_relative_error'],
                                          old_stop=b['stop'], new_stop=a['stop']))
    out['amend2_unchanged_paths_identical'] = all(r['identical'] for r in out['amend2'] if r['failed_trials'] == 0)
    out['amend2_changed_runs'] = [r for r in out['amend2'] if r['failed_trials'] > 0]
    # Amendment 3: NC-exact (exact parity evaluator) at seeds 10-14 must equal the Phase-1 evaluator control.
    out['nc_exact'] = []
    P1E = ROOT / 'numerical-section-revision' / 'phase1-2026-11-11' / 'runs' / 'evaluator'
    for d in (5, 10, 20):
        for g in (0, 1):
            for seed in range(10, 15):
                pa = BASE / 'cohort_pb/results_nc_exact' / f'pb_d{d}_g{g}_s{seed}_NCx.json'
                pb = P1E / f'pb_NC_analytic_d{d}_g{g}_s{seed}.json'
                if not (pa.exists() and pb.exists()):
                    continue
                a = json.loads(pa.read_text()); b = json.loads(pb.read_text())
                eq = {k: a[k] == b[k] for k in ('final_relative_error', 'fits', 'iterations', 'stop', 'coefficients', 'blocks')}
                out['nc_exact'].append(dict(dimension=d, geometry=g, seed=seed, identical=all(eq.values()), fields=eq))
    out['nc_exact_identical_to_phase1'] = all(r['identical'] for r in out['nc_exact']) if out['nc_exact'] else None
    out['nc_exact_checked'] = len(out['nc_exact'])
    out['cohort1d_all_identical'] = all(r['identical'] for r in out['cohort1d'])
    out['cohort_pb_NC_C_T_all_identical'] = all(r['identical'] for r in out['cohort_pb'] if r['method'] != 'S')
    out['cohort_pb_S_identical_where_expected'] = all(r['identical'] for r in out['cohort_pb']
                                                      if r['method'] == 'S' and r['expected_identical'])
    out['cohort_pb_S_changed'] = [r for r in out['cohort_pb'] if r['method'] == 'S' and not r['identical']]
    (HERE / 'reproduction.json').write_text(json.dumps(out, indent=2) + '\n')
    print({k: v for k, v in out.items() if k not in ('cohort1d', 'cohort_pb', 'amend2', 'amend2_changed_runs', 'cohort_pb_S_changed', 'nc_exact')})
    print('amendment-2 runs with failed trials:', len(out['amend2_changed_runs']))


if __name__ == '__main__':
    main()
