#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): summary of the evaluator control in the full-span linear family and the 2D controls.

For every start: C (pairs in t, exact parity evaluator), NC and C_d with the frozen eight-node averaged evaluator
(guarded records), and NC and C_d with the exact evaluator (exact_evaluator_runner.py).  Writes
evaluator-control.json with medians, success counts (E <= 1e-6) and median fits per cohort, and every run."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / 'guarded-rerun-2026-11-09' / 'first-order' / 'accuracy'
SUPP = HERE.parent / 'guarded-rerun-2026-11-09' / 'supplementary-2d'
METHODS = ('NC_avg', 'NC_exact', 'C', 'Cd_avg', 'Cd_exact')


def rec(path):
    r = json.loads(Path(path).read_text())
    return dict(E=r['final_relative_error'], fits=r['fits'], stop=r.get('stop'))


def summary(rows):
    out = {}
    for m in METHODS:
        have = [r[m] for r in rows if r.get(m)]
        if not have:
            continue
        E = [x['E'] for x in have]
        out[m] = dict(n=len(have), median_E=st.median(E), s6=sum(e <= 1e-6 for e in E),
                      s10=sum(e <= 1e-10 for e in E), min_E=min(E), max_E=max(E),
                      median_fits=st.median([x['fits'] for x in have]))
    both = [r for r in rows if r.get('NC_exact') and r.get('C')]
    out['C_below_NC_exact'] = sum(r['C']['E'] < r['NC_exact']['E'] for r in both)
    out['NC_exact_below_NC_avg'] = sum(r['NC_exact']['E'] < r['NC_avg']['E'] for r in rows
                                       if r.get('NC_exact') and r.get('NC_avg'))
    out['n_pairs_compared'] = len(both)
    return out


def main():
    res = {}
    rows = []
    for d in (10, 20):
        for s in (10, 11, 12):
            row = dict(d=d, seed=s)
            for m, p in (('NC_avg', FIRST / f'linear_d{d}_g0_s{s}_stable.json'),
                         ('Cd_avg', FIRST / f'linear_d{d}_g0_s{s}_adaptive.json'),
                         ('C', HERE / 'runs' / 'fullspan' / f'linear_d{d}_g0_s{s}_T.json'),
                         ('NC_exact', HERE / 'runs' / 'evaluator-linear' / f'linear_stable_exact_d{d}_g0_s{s}.json'),
                         ('Cd_exact', HERE / 'runs' / 'evaluator-linear' / f'linear_adaptive_exact_d{d}_g0_s{s}.json')):
                if p.exists():
                    row[m] = rec(p)
            rows.append(row)
    res['fullspan'] = {f'd{d}': summary([r for r in rows if r['d'] == d]) for d in (10, 20)}
    res['fullspan_rows'] = rows
    tasks = [t for t in json.loads((SUPP / 'tasks.json').read_text())
             if t['method'] == 'stable' and t['energy_factor'] is None and t['cohort'] in
             ('recovery', 'mismatch001', 'mismatch005', 'pilot-smoke', 'pilot-smooth', 'pilot-multi6')]
    rows2 = []
    for t in tasks:
        base = t['id'][:-len('_stable')]
        row = dict(id=base, cohort=t['cohort'], case=t['case'])
        for m, p in (('NC_avg', SUPP / 'accuracy' / f'{base}_stable.json'),
                     ('Cd_avg', SUPP / 'accuracy' / f'{base}_adaptive.json'),
                     ('C', HERE / 'runs' / 'twod' / f'{base}_stable_T.json'),
                     ('NC_exact', HERE / 'runs' / 'evaluator-twod' / f'{base}_stable_exact.json'),
                     ('Cd_exact', HERE / 'runs' / 'evaluator-twod' / f'{base}_adaptive_exact.json')):
            if p.exists():
                row[m] = rec(p)
        rows2.append(row)
    groups = dict(recovery=['recovery'], mismatch001=['mismatch001'], mismatch005=['mismatch005'],
                  forcing_smooth=['pilot-smooth'], forcing_other=['pilot-multi6', 'pilot-smoke'])
    res['twod'] = {g: summary([r for r in rows2 if r['cohort'] in c]) for g, c in groups.items()}
    for g, c in groups.items():                      # largest relative difference between methods, per start
        diffs = []
        for r in rows2:
            if r['cohort'] in c and all(r.get(m) for m in METHODS):
                E = [r[m]['E'] for m in METHODS]
                diffs.append(max(E) / min(E) - 1)
        res['twod'][g]['max_rel_spread_all_methods'] = max(diffs) if diffs else None
    res['twod_rows'] = rows2
    # Directional solver: NC with the eight-node formulas (records of the paper) against NC with the series of C
    # below t = 5e-6 sigma^2 (chain_series_runner.py), and C.
    DIR = HERE.parent / 'guarded-rerun-2026-11-09' / 'directional'
    rows3 = []
    for problem, seeds, nc_dir in (('linear1d', range(0, 10), HERE / 'runs' / 'linear1d'),
                                   ('second1d', range(40, 50), DIR / 'accuracy-second1d')):
        for s in seeds:
            row = dict(problem=problem, seed=s)
            for m, p in (('NC_avg', nc_dir / f'{problem}_d1_g0_seed{s}_stable.json'),
                         ('NC_exact', HERE / 'runs' / 'evaluator-chain' / f'{problem}_seed{s}_stable_series.json'),
                         ('C', HERE / 'runs' / problem / f'{problem}_d1_g0_seed{s}_tpair.json')):
                if p.exists():
                    row[m] = rec(p)
            rows3.append(row)
    res['chain'] = {p: summary([r for r in rows3 if r['problem'] == p]) for p in ('linear1d', 'second1d')}
    for p in ('linear1d', 'second1d'):
        rs = [r for r in rows3 if r['problem'] == p and r.get('NC_exact') and r.get('NC_avg')]
        res['chain'][p]['median_ratio_NC_exact_over_NC_avg'] = (st.median([r['NC_exact']['E'] / r['NC_avg']['E'] for r in rs])
                                                               if rs else None)
    res['chain_rows'] = rows3
    (HERE / 'evaluator-control.json').write_text(json.dumps(res, indent=1))
    for k in ('fullspan', 'twod', 'chain'):
        print(k, json.dumps(res[k], indent=1))


if __name__ == '__main__':
    main()
