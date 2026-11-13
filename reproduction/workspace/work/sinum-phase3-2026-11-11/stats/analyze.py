#!/usr/bin/env python3
"""Pre-registered analysis of the Phase-3 cohorts (see ../prereg/PREREGISTRATION.md).

Writes per-start CSV files, summary.json (medians, success counts, exact McNemar
tests, Wilcoxon signed-rank tests, sign-test intervals, Holm-adjusted primary
p-values, fit comparisons) and table-rows.tex. Uses every run, including
work-limit, abnormal and failed returns.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
THRESHOLDS = [1e-4, 1e-6, 1e-8, 1e-10]


# File suffix -> method label (amendment 1): C is the paper's method (pairs in t = h^2), Cd the
# discrete-completion ablation (former C).
LABELS_1D = dict(S='S', NC='NC', C='Cd', tpair='C')            # pre-amendment-2 records (cohort1d/results)
LABELS_1D_A2 = dict(S='S', NC='NC', Cd='Cd', C='C')             # amendment 2 (cohort1d/results_amend2): primary
# Cohort B after amendment 3: NC and C_d with the exact pair evaluator of C are the main arms; the frozen
# averaged-evaluator runs are NCa and Cda (secondary).
LABELS_PB = dict(S='S', NC='NCa', C='Cda', T='C')
EXTRA_PB = (('results_nc_exact', 'NCx', 'NC'), ('results_cd_exact', 'Cdx', 'Cd'))


def load_1d(directory='results_amend2'):
    rows = []
    labels = LABELS_1D_A2 if directory == 'results_amend2' else LABELS_1D
    for order in (2, 3):
        for seed in range(900, 1000):
            for suffix, m in labels.items():
                p = BASE / 'cohort1d' / directory / f'order{order}_seed{seed}_{suffix}.json'
                if suffix == 'tpair' and not p.exists():
                    continue
                r = json.loads(p.read_text())
                rows.append(dict(stratum=f'r={order}', order=order, seed=seed, method=m,
                                 error=r['final_relative_error'], fits=r['fits'], stop=r['stop'],
                                 completions=r.get('completions', 0), joins=r.get('joins', 0),
                                 failed_trials=sum(v for k, v in (r.get('guard_counts') or {}).items()
                                                   if str(k).startswith('optimizer_trial:'))))
    return rows


def load_pb():
    rows = []
    for d in (5, 10, 20):
        for g in (0, 1):
            for seed in range(10, 35):
                for sub, suffix, m in [('results', k, v) for k, v in LABELS_PB.items()] + list(EXTRA_PB):
                    p = BASE / 'cohort_pb' / sub / f'pb_d{d}_g{g}_s{seed}_{suffix}.json'
                    if not p.exists():                     # incomplete arm: reported with its actual n
                        continue
                    r = json.loads(p.read_text())
                    rows.append(dict(stratum=f'd={d}', dimension=d, geometry=g, seed=seed, method=m,
                                     error=r['final_relative_error'], fits=r['fits'], stop=r['stop'],
                                     completions=r.get('completions', 0),
                                     zero_gaps=r.get('zero_squared_gaps') if suffix == 'T' else None,
                                     energy_gap=r.get('energy_gap')))
    return rows


def paired(rows, stratum, a, b, key):
    """Matched arrays of method a and method b in a stratum (None = pooled)."""
    sel = [r for r in rows if stratum is None or r['stratum'] == stratum]
    da = {key(r): r for r in sel if r['method'] == a}
    db = {key(r): r for r in sel if r['method'] == b}
    keys = sorted(set(da) & set(db))
    return [da[k] for k in keys], [db[k] for k in keys]


def mcnemar_exact(sa, sb):
    """Two-sided exact McNemar test; sa, sb boolean success arrays of the two methods."""
    b = int(np.sum(sa & ~sb))      # a succeeds, b fails
    c = int(np.sum(~sa & sb))      # a fails, b succeeds
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, .5))
    return dict(a_only=b, b_only=c, both=int(np.sum(sa & sb)), neither=int(np.sum(~sa & ~sb)), p=p)


def sign_interval(diff, level=.95):
    """Median of paired differences with its distribution-free interval (order statistics)."""
    d = np.sort(np.asarray(diff))
    n = len(d)
    alpha = 1 - level
    k = 0
    while stats.binom.cdf(k, n, .5) <= alpha / 2:
        k += 1
    # k is the smallest index with cdf > alpha/2; interval [d_(k), d_(n-k+1)] in 1-based order statistics
    lo, hi = d[k - 1] if k >= 1 else -math.inf, d[n - k] if k >= 1 else math.inf
    coverage = 1 - 2 * stats.binom.cdf(k - 1, n, .5) if k >= 1 else 1.0
    return dict(median=float(np.median(d)), lower=float(lo), upper=float(hi), coverage=float(coverage))


def compare(rows, stratum, a, b, key):
    ra, rb = paired(rows, stratum, a, b, key)
    if not ra:
        return dict(pair=f'{a}-{b}', stratum=stratum or 'pooled', n=0)
    ea = np.array([r['error'] for r in ra])
    eb = np.array([r['error'] for r in rb])
    la, lb = np.log10(np.maximum(ea, 1e-300)), np.log10(np.maximum(eb, 1e-300))
    diff = la - lb
    out = dict(pair=f'{a}-{b}', stratum=stratum or 'pooled', n=len(ra))
    out['mcnemar'] = {f'{t:g}': mcnemar_exact(ea <= t, eb <= t) for t in THRESHOLDS}
    nonzero = diff[diff != 0]
    if len(nonzero):
        w = stats.wilcoxon(la, lb, zero_method='wilcox', alternative='two-sided')
        out['wilcoxon'] = dict(statistic=float(w.statistic), p=float(w.pvalue), nonzero=int(len(nonzero)),
                               a_lower=int(np.sum(diff < 0)), b_lower=int(np.sum(diff > 0)))
    else:
        out['wilcoxon'] = dict(statistic=None, p=1.0, nonzero=0, a_lower=0, b_lower=0)
    out['log10_ratio'] = sign_interval(diff)
    fa = np.array([r['fits'] for r in ra]); fb = np.array([r['fits'] for r in rb])
    fw = stats.wilcoxon(fa, fb, zero_method='wilcox') if np.any(fa != fb) else None
    out['fits'] = dict(median_a=float(np.median(fa)), median_b=float(np.median(fb)),
                       median_difference=float(np.median(fa - fb)),
                       wilcoxon_p=float(fw.pvalue) if fw is not None else 1.0)
    return out


def describe(rows, stratum, method):
    sel = [r for r in rows if r['method'] == method and (stratum is None or r['stratum'] == stratum)]
    if not sel:
        return dict(method=method, stratum=stratum or 'pooled', n=0)
    e = np.array([r['error'] for r in sel])
    f = np.array([r['fits'] for r in sel])
    stops = {}
    for r in sel:
        stops[r['stop']] = stops.get(r['stop'], 0) + 1
    out = dict(method=method, stratum=stratum or 'pooled', n=len(sel), median_error=float(np.median(e)),
               quartiles=[float(np.quantile(e, .25)), float(np.quantile(e, .75))],
               min_error=float(e.min()), max_error=float(e.max()),
               successes={f'{t:g}': int(np.sum(e <= t)) for t in THRESHOLDS},
               median_fits=float(np.median(f)), total_fits=int(f.sum()), stops=stops,
               completions=int(sum(r['completions'] or 0 for r in sel)))
    if method == 'C' and sel[0].get('zero_gaps') is not None:
        out['zero_gap_pairs'] = int(sum(r['zero_gaps'] for r in sel))
        out['runs_with_zero_gap'] = int(sum(r['zero_gaps'] > 0 for r in sel))
    return out


def holm(pvalues):
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1., running)
    return adj.tolist()


def write_csv(path, rows, keys):
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    rows1 = load_1d()
    rowsb = load_pb()
    write_csv(HERE / 'cohort1d-per-run.csv', rows1, ['order', 'seed', 'method', 'error', 'fits', 'stop', 'completions', 'joins',
                                                     'failed_trials'])
    rows1_pre = load_1d('results')
    write_csv(HERE / 'cohort1d-per-run-pre-amendment2.csv', rows1_pre, ['order', 'seed', 'method', 'error', 'fits', 'stop',
                                                                         'completions', 'joins'])
    write_csv(HERE / 'cohort-pb-per-run.csv', rowsb, ['dimension', 'geometry', 'seed', 'method', 'error', 'fits', 'stop',
                                                      'completions', 'zero_gaps', 'energy_gap'])
    key1 = lambda r: (r['order'], r['seed'])
    keyb = lambda r: (r['dimension'], r['geometry'], r['seed'])
    summary = dict(thresholds=THRESHOLDS, cohort1d=dict(describe=[], compare=[]), cohort_pb=dict(describe=[], compare=[]))
    have_c1 = any(r['method'] == 'C' for r in rows1)
    pairs1 = ((('C', 'NC'), ('Cd', 'NC'), ('C', 'Cd'), ('C', 'S'), ('NC', 'S'), ('Cd', 'S')) if have_c1
              else (('Cd', 'NC'), ('NC', 'S'), ('Cd', 'S')))
    summary['cohort1d']['new_C_available'] = have_c1
    summary['cohort1d_pre_amendment2'] = dict(
        describe=[describe(rows1_pre, st, m) for st in ('r=2', 'r=3') for m in ('S', 'NC', 'C', 'Cd')],
        compare=[compare(rows1_pre, st, a, b, key1) for st in ('r=2', 'r=3')
                 for a, b in (('C', 'NC'), ('Cd', 'NC'), ('C', 'Cd'))])
    summary['cohort1d']['failed_trials'] = {f"{st} {m}": sum(r['failed_trials'] > 0 for r in rows1 if r['stratum'] == st and r['method'] == m)
                                            for st in ('r=2', 'r=3') for m in ('S', 'NC', 'C', 'Cd')}
    for st in ('r=2', 'r=3'):
        for m in ('S', 'NC', 'C', 'Cd'):
            summary['cohort1d']['describe'].append(describe(rows1, st, m))
        for a, b in pairs1:
            summary['cohort1d']['compare'].append(compare(rows1, st, a, b, key1))
    for st in ('d=5', 'd=10', 'd=20', None):
        for m in ('S', 'NC', 'C', 'Cd', 'NCa', 'Cda'):
            summary['cohort_pb']['describe'].append(describe(rowsb, st, m))
        for a, b in (('C', 'NC'), ('Cd', 'NC'), ('C', 'Cd'), ('C', 'S'), ('NC', 'S'), ('Cd', 'S'),
                     ('NC', 'NCa'), ('Cd', 'Cda'), ('C', 'NCa'), ('C', 'Cda')):
            summary['cohort_pb']['compare'].append(compare(rowsb, st, a, b, keyb))
    # Primary families and Holm adjustment
    fam1 = []
    primary1 = 'C-NC' if have_c1 else 'Cd-NC'
    summary['cohort1d']['primary_pair'] = primary1
    for c in summary['cohort1d']['compare']:
        if c['pair'] == primary1:
            fam1 += [(c, 'mcnemar', '1e-06'), (c, 'mcnemar', '1e-10'), (c, 'wilcoxon', None)]
    famb = []
    primaryb = 'C-NC'          # amendment 3: NC with the exact pair evaluator of C
    summary['cohort_pb']['primary_pair'] = primaryb
    for c in summary['cohort_pb']['compare']:
        if c['pair'] == primaryb:
            famb += [(c, 'mcnemar', '1e-06'), (c, 'wilcoxon', None)]
    fam1 = [x for x in fam1 if x[0].get('n')]
    famb = [x for x in famb if x[0].get('n')]
    for fam, name in ((fam1, 'cohort1d'), (famb, 'cohort_pb')):
        ps = [(c['mcnemar'][t]['p'] if kind == 'mcnemar' else c['wilcoxon']['p']) for c, kind, t in fam]
        adj = holm(ps)
        summary[name]['primary'] = [dict(stratum=c['stratum'], test=kind, threshold=t, p=p, holm=q)
                                    for (c, kind, t), p, q in zip(fam, ps, adj)]
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['cohort1d']['primary'], indent=1))
    print(json.dumps(summary['cohort_pb']['primary'], indent=1))


if __name__ == '__main__':
    main()
