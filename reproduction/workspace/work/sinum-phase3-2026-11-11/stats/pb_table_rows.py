#!/usr/bin/env python3
"""Rows of the recovery table (Table 3) for the first-order PB family (n = 50 per dimension) and its C_d sub-table."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'summary.json').read_text())


def sci2(x):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f'${m:.1f}\\times10^{{{e}}}$'


def find(**kw):
    for it in S['cohort_pb']['describe']:
        if all(it.get(k) == v for k, v in kw.items()):
            return it


main, abl = [], []
for d in (5, 10, 20):
    st = f'd={d}'
    s_, nc, c, cd = (find(method=m, stratum=st) for m in ('S', 'NC', 'C', 'Cd'))
    n = nc['n']
    assert c['n'] == n == cd['n']
    s_ok = s_ and s_.get('n') == n
    s_med = sci2(s_['median_error']) if s_ok else '@S@'
    s_suc = str(s_['successes']['1e-06']) if s_ok else '@S@'
    s_fit = f"{s_['median_fits']:g}" if s_ok else '@S@'
    main.append(f"PB, $d={d}$, $r=1$ & {n} & {s_med} & {sci2(nc['median_error'])} & {sci2(c['median_error'])} & "
                f"{s_suc} & {nc['successes']['1e-06']} & {c['successes']['1e-06']} & {s_fit} & {nc['median_fits']:g} & {c['median_fits']:g}\\\\")
    abl.append(f"PB, $d={d}$, $r=1$ & {n} & {sci2(cd['median_error'])} & {cd['successes']['1e-06']} & {cd['median_fits']:g}\\\\")
print('\n'.join(main))
print()
print('\n'.join(abl))
