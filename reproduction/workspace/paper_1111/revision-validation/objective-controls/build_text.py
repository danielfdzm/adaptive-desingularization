#!/usr/bin/env python3
"""Produce insertion-ready LaTeX from the current objective-control records."""
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def sci(x,digits=3):
    if x==0:return '$0$'
    exponent=math.floor(math.log10(abs(x)))
    return f'${x/10**exponent:.{digits-1}f}\\times10^{{{exponent}}}$'


def main():
    data=json.loads((HERE/'results.json').read_text())
    records=data['records']
    def rows(family,objective='residual',chart='t',floor=0.):
        return [r for r in records if (r['family'],r['objective'],r['chart'],r['t_floor'])==
                (family,objective,chart,floor)]
    def med(group,key):return float(np.median([r[key] for r in group]))
    table=[r'\begin{table}[htbp]\centering\footnotesize',
           r'\setlength{\tabcolsep}{3pt}',
           r'\caption{Exploratory local objective controls, ten matched starts per target. G uses the shifted Gram objective and chart-specific stopping; R uses the directly evaluated residual objective and common feasible $(\beta,t)$ residual stopping. Errors and canonical gradients are independently checked using 80-digit correlations; projected residuals are evaluated in double precision on the common domain $t\ge0$. Every entry is a median. These small linear problems are mechanism controls.}',
           r'\label{tab:objective-controls}',
           r'\begin{tabular}{@{}llrrrrrr@{}}\toprule',
           r'Target & Objective & $E_h$ & $E_t$ & $\|R_t\|_h$ & $\|R_t\|_t$ & Fits $h$ & Fits $t$\\\midrule']
    labels={'zero_gap':"$G'$",'positive_gap':"$G_{0.99}'$",'ordinary':r'$G(\cdot-1)+G(\cdot+1)$'}
    for family in labels:
        for objective,label in [('shifted_gram','G'),('residual','R')]:
            H,T=rows(family,objective,'h',1e-24),rows(family,objective)
            table.append(' & '.join([labels[family],label,sci(med(H,'error')),sci(med(T,'error')),
                                     sci(med(H,'canonical_residual')),sci(med(T,'canonical_residual')),
                                     f"{med(H,'fits'):g}",f"{med(T,'fits'):g}"])+r'\\')
    table.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (HERE/'table-objective-controls.tex').write_text('\n'.join(table)+'\n')
    floor_table=[r'\begin{table}[htbp]\centering\footnotesize',
                 r'\caption{Positive-floor controls with the residual objective, ten starts per row. $t\ge\varepsilon$ permits only ordinary pairs for $\varepsilon>0$. The residual is assessed in the common domain $t\ge0$, even when the iteration stops at its positive floor. The last column reconstructs the ordinary Gaussian coefficients for the zero-gap target. All entries are medians.}',
                 r'\label{tab:positive-floor-controls}',
                 r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
                 " & \\multicolumn{2}{c}{$G'$} & \\multicolumn{2}{c}{$G_{0.99}'$} & \\\\",
                 r'$\varepsilon$ & $E$ & $\|R_t\|$ & $E$ & $\|R_t\|$ & Ordinary weight norm\\\midrule']
    for floor in (0.,1e-4,1e-8,1e-12,1e-24):
        Z,P=rows('zero_gap',floor=floor),rows('positive_gap',floor=floor)
        norms=[math.sqrt((r['coefficients'][0]**2+r['coefficients'][1]**2/r['t'])/2) for r in Z if r['t']>0]
        norm='---' if floor==0 else sci(float(np.median(norms)))
        floor_table.append(' & '.join([sci(floor),sci(med(Z,'error')),sci(med(Z,'canonical_residual')),
                                      sci(med(P,'error')),sci(med(P,'canonical_residual')),norm])+r'\\')
    floor_table.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (HERE/'table-positive-floors.tex').write_text('\n'.join(floor_table)+'\n')
    times=[r'\begin{table}[htbp]\centering\footnotesize',
           r'\caption{Descriptive serial timings of the residual-objective controls: median milliseconds over ten starts, one run per start and method, one numerical-library thread. Project numerical jobs were serialized; unrelated host activity was not controlled. Independent endpoint validation is excluded.}',
           r'\label{tab:objective-control-times}',
           r'\begin{tabular}{@{}lrr@{}}\toprule Target & $h$ & $t\ge0$\\\midrule']
    for family in labels:
        H,T=rows(family,'residual','h',1e-24),rows(family)
        times.append(f"{labels[family]} & {1000*med(H,'seconds'):.2f} & {1000*med(T,'seconds'):.2f}"+r'\\')
    times.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (HERE/'table-objective-timing.tex').write_text('\n'.join(times)+'\n')
    print('Wrote three LaTeX tables directly from results.json.')


if __name__=='__main__':main()
