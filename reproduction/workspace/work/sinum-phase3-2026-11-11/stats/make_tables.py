#!/usr/bin/env python3
"""LaTeX tables from summary.json: the compact main-text paired table and the supplement tables SM1-SM3.

Writes table-paired-main.tex, table-sm-describe-1d.tex, table-sm-describe-pb.tex, table-sm-compare.tex.
Numbers are formatted only; every value comes from summary.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LBL = dict(S='S', NC='NC', NCa=r'NC$_{\rm a}$', C='C', Cd=r'C$_{\rm d}$', Cda=r'C$_{\rm d,a}$')


def sci(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '--'
    if x == 0:
        return '$0$'
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    if -2 <= e <= 0:
        return f'${x:.{digits - 1 - e}f}$'.replace('.0$', '$') if e < 0 else f'${x:.{digits - 1}f}$'
    return f'${m:.{digits - 1}f}\\times10^{{{e}}}$'


def pval(p):
    if p >= .1:
        return f'${p:.2f}$'
    if p >= 1e-3:
        return f'${p:.3f}$'
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f'${m:.1f}\\times10^{{{e}}}$'


def ratio(lr):
    """10^median of log10 ratios with interval, as a power of ten."""
    f = lambda v: '-\\infty' if v == -math.inf else ('\\infty' if v == math.inf else f'{v:.1f}'.replace('-0.0', '0.0'))
    return f"$10^{{{f(lr['median'])}}}$ $[10^{{{f(lr['lower'])}}},10^{{{f(lr['upper'])}}}]$"


def longtable_body(head, rows):
    """Header repeated on every page, then the rows; the caller supplies the longtable environment."""
    return '\n'.join([r'\toprule'] + head + [r'\midrule\endfirsthead', r'\toprule'] + head +
                      [r'\midrule\endhead', r'\bottomrule\endfoot'] + rows) + '\n'


def find(items, **kw):
    for it in items:
        if all(it.get(k) == v for k, v in kw.items()):
            return it
    return None


def main():
    S = json.loads((HERE / 'summary.json').read_text())
    pair1 = S['cohort1d'].get('primary_pair', 'C-NC')
    a1 = pair1.split('-')[0]
    # Main-text paired table: the paper's C (or the ablation if C is missing in cohort A) against NC
    rows = []
    for st, lab in (('r=2', 'Independent, $r=2$'), ('r=3', 'Independent, $r=3$')):
        c = find(S['cohort1d']['compare'], pair=pair1, stratum=st)
        dn = find(S['cohort1d']['describe'], method='NC', stratum=st)
        dc = find(S['cohort1d']['describe'], method=a1, stratum=st)
        rows.append((lab, c['n'], dn, dc, c))
    for st, lab in (('d=5', 'PB $r=1$, $d=5$'), ('d=10', 'PB $r=1$, $d=10$'), ('d=20', 'PB $r=1$, $d=20$'),
                    ('pooled', 'PB $r=1$, pooled')):
        pb_pair = S['cohort_pb'].get('primary_pair', 'C-NC')
        nc_label = pb_pair.split('-')[1]
        c = find(S['cohort_pb']['compare'], pair=pb_pair, stratum=st)
        dn = find(S['cohort_pb']['describe'], method=nc_label, stratum=st)
        dc = find(S['cohort_pb']['describe'], method='C', stratum=st)
        rows.append((lab, c['n'], dn, dc, c))
    out = [r'\begin{tabular}{@{}lr cc r cc r r l@{}}', r'\toprule',
           r' & & \multicolumn{3}{c}{$E\le10^{-6}$} & \multicolumn{3}{c}{$E\le10^{-10}$} & Wilcoxon & Median $E_{\rm C}/E_{\rm NC}$\\',
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
           r'Study & $n$ & NC & C & $p$ & NC & C & $p$ & $p$ & [95\% interval]\\', r'\midrule']
    for lab, n, dn, dc, c in rows:
        m6, m10 = c['mcnemar']['1e-06'], c['mcnemar']['1e-10']
        out.append(f"{lab} & {n} & {dn['successes']['1e-06']} & {dc['successes']['1e-06']} & {pval(m6['p'])} & "
                   f"{dn['successes']['1e-10']} & {dc['successes']['1e-10']} & {pval(m10['p'])} & "
                   f"{pval(c['wilcoxon']['p'])} & {ratio(c['log10_ratio'])}\\\\")
    out += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'table-paired-main.tex').write_text('\n'.join(out) + '\n')
    # Supplement: descriptive tables
    for name, coh, strata, methods in (('1d', 'cohort1d', ('r=2', 'r=3'), ('S', 'NC', 'C', 'Cd')),
                                       ('pb', 'cohort_pb', ('d=5', 'd=10', 'd=20', 'pooled'), ('S', 'NC', 'C', 'Cd', 'NCa', 'Cda'))):
        out = [r'\begin{tabular}{@{}llr ccc rrrr r@{}}', r'\toprule',
               r'Stratum & Method & $n$ & Median $E$ & \multicolumn{2}{c}{Quartiles} & $10^{-4}$ & $10^{-6}$ & $10^{-8}$ & $10^{-10}$ & Fits\\',
               r'\midrule']
        for st in strata:
            first = True
            for m in methods:
                d = find(S[coh]['describe'], method=m, stratum=st)
                if not d or not d.get('n'):
                    continue
                q = d['quartiles']
                sc = d['successes']
                stl = ('$' + st.replace('=', '=') + '$') if st != 'pooled' else 'pooled'
                out.append(f"{stl if first else ''} & {LBL[m]} & {d['n']} & {sci(d['median_error'])} & {sci(q[0])} & {sci(q[1])} & "
                           f"{sc['0.0001']} & {sc['1e-06']} & {sc['1e-08']} & {sc['1e-10']} & {d['median_fits']:g}\\\\")
                first = False
            out.append(r'\midrule')
        out[-1] = r'\bottomrule'
        out.append(r'\end{tabular}')
        (HERE / f'table-sm-describe-{name}.tex').write_text('\n'.join(out) + '\n')
    # Supplement: all paired comparisons
    out = [r'\begin{tabular}{@{}llr cccc r l r@{}}', r'\toprule',
           r'Stratum & Pair $A$--$B$ & $n$ & \multicolumn{4}{c}{Exact McNemar: $A$ only / $B$ only ($p$)} & Wilcoxon & Median $E_A/E_B$ & $\Delta$ fits\\',
           r'\cmidrule(lr){4-7}',
           r' & & & $10^{-4}$ & $10^{-6}$ & $10^{-8}$ & $10^{-10}$ & $p$ & [95\% interval] & \\', r'\midrule']
    for coh in ('cohort1d', 'cohort_pb'):
        for c in S[coh]['compare']:
            if not c.get('n'):
                continue
            a, b = c['pair'].split('-')
            cells = []
            for t in ('0.0001', '1e-06', '1e-08', '1e-10'):
                m = c['mcnemar'][t]
                cells.append(f"{m['a_only']}/{m['b_only']} ({pval(m['p'])})")
            st = c['stratum'] if c['stratum'] == 'pooled' else '$' + c['stratum'] + '$'
            st = ('PB ' if coh == 'cohort_pb' else 'Indep. ') + st
            out.append(f"{st} & {LBL[a]}--{LBL[b]} & {c['n']} & " + ' & '.join(cells) +
                       f" & {pval(c['wilcoxon']['p'])} & {ratio(c['log10_ratio'])} & {c['fits']['median_difference']:+g}\\\\")
        out.append(r'\midrule')
    out[-1] = r'\bottomrule'
    out.append(r'\end{tabular}')
    (HERE / 'table-sm-compare.tex').write_text('\n'.join(out) + '\n')
    (HERE / 'table-sm-compare-longtable.tex').write_text(longtable_body(out[2:5], out[6:-2]))
    # Supplement: amendment 2 (rejected trials) and the pre-amendment one-dimensional cohort
    rep = json.loads((HERE / 'reproduction.json').read_text()) if (HERE / 'reproduction.json').exists() else {}
    changed = rep.get('amend2_changed_runs', [])
    if changed:
        out = [r'\begin{tabular}{@{}rrl ll@{}}', r'\toprule',
               r'$r$ & Seed & Method & Frozen rule: $E$ (stop) & Revised rule: $E$ (stop)\\', r'\midrule']
        stopname = lambda x: {'function_tolerance': 'tolerance', 'singular unconstrained coefficient system': 'singular solve',
                              'unreliable unconstrained coefficient solve': 'unreliable solve', 'abnormal': 'abnormal',
                              'work_limit': 'fit limit'}.get(x, x)
        for c in sorted(changed, key=lambda c: (c['order'], c['seed'], c['method'])):
            out.append(f"{c['order']} & {c['seed']} & {LBL[c['method']]} & {sci(c['old_error'])} ({stopname(c['old_stop'])}) & "
                       f"{sci(c['new_error'])} ({stopname(c['new_stop'])})\\\\")
        out += [r'\bottomrule', r'\end{tabular}']
        (HERE / 'table-sm-amend2.tex').write_text('\n'.join(out) + '\n')
        (HERE / 'table-sm-amend2-longtable.tex').write_text(longtable_body(out[2:3], out[4:-2]))
    pre = S.get('cohort1d_pre_amendment2')
    if pre:
        out = [r'\begin{tabular}{@{}llr crrr r@{}}', r'\toprule',
               r'$r$ & Method & $n$ & Median $E$ & $10^{-6}$ & $10^{-10}$ & Aborts & Fits\\', r'\midrule']
        for st in ('r=2', 'r=3'):
            first = True
            for m in ('S', 'NC', 'C', 'Cd'):
                d = find(pre['describe'], method=m, stratum=st)
                if not d or not d.get('n'):
                    continue
                ab = sum(v for k, v in d['stops'].items() if 'coefficient' in k)
                out.append(f"{st[2:] if first else ''} & {LBL[m]} & {d['n']} & {sci(d['median_error'])} & "
                           f"{d['successes']['1e-06']} & {d['successes']['1e-10']} & {ab} & {d['median_fits']:g}\\\\")
                first = False
            out.append(r'\midrule')
        out[-1] = r'\bottomrule'
        out.append(r'\end{tabular}')
        (HERE / 'table-sm-pre-amend2.tex').write_text('\n'.join(out) + '\n')
    # Supplement: stationarity residuals (plan step 7 / Section 4.3), from stationarity/stationarity-rows.json
    stp = HERE.parent / 'stationarity' / 'stationarity-rows.json'
    if stp.exists():
        rows = json.loads(stp.read_text())['rows']
        order = [('1D independent', m) for m in ('S', 'NC', 'C', 'C_d')] + \
                [('PB first order', m) for m in ('S', 'NC', 'C', 'C_d', 'NC_averaged', 'C_d_averaged')]
        name = {'NC_averaged': r'NC$_{\rm a}$', 'C_d_averaged': r'C$_{\rm d,a}$', 'C_d': r'C$_{\rm d}$'}
        out = [r'\begin{tabular}{@{}llr cc rrr@{}}', r'\toprule',
               r'Study & Method & $n$ & Median $R_0$ & Max $R_0$ & \multicolumn{3}{c}{Cleanup exit}\\',
               r'\cmidrule(l){6-8}', r' & & & & & tolerance & rounding & work limit\\', r'\midrule']
        for coh, m in order:
            sel = [r for r in rows if r['cohort'] == coh and r['method'] == m]
            if not sel:
                continue
            r0 = sorted(r['R0'] for r in sel)
            assert len(r0) == len(sel), 'missing R0'
            med = r0[len(r0) // 2] if len(r0) % 2 else (r0[len(r0) // 2 - 1] + r0[len(r0) // 2]) / 2
            if coh.startswith('1D'):
                ex = {k: sum(r['cleanup_stop'] == k for r in sel) for k in ('tolerance', 'rounding', 'work_limit')}
                assert sum(ex.values()) == len(sel), 'unexpected cleanup exit'
                cells = f"{ex['tolerance']} & {ex['rounding']} & {ex['work_limit']}"
            else:
                cells = r'\multicolumn{3}{c}{no cleanup}'
            out.append(f"{'Independent' if coh.startswith('1D') else 'PB, $r=1$'} & {name.get(m, m)} & {len(sel)} & "
                       f"{sci(med)} & {sci(max(r0))} & {cells}\\\\")
        out += [r'\bottomrule', r'\end{tabular}']
        (HERE / 'table-sm-stationarity.tex').write_text('\n'.join(out) + '\n')
    prim = []
    for coh in ('cohort1d', 'cohort_pb'):
        for p in S[coh].get('primary', []):
            prim.append(f"{coh} {p['stratum']} {p['test']} {p['threshold']}: p={p['p']:.3g} Holm={p['holm']:.3g}")
    (HERE / 'primary-tests.txt').write_text('\n'.join(prim) + '\n')
    print('\n'.join(prim))


if __name__ == '__main__':
    main()
