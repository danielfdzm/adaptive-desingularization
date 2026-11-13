"""Summarize every frozen validation outcome, including failures."""
import csv,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    summary=[]
    for suite in ['validation']:
        rr=[]
        for f in sorted((HERE/('unconstrained-'+suite)).glob('*.json')):
            r=json.loads(f.read_text())
            if isinstance(r,dict) and 'seed' in r and 'method' in r:rr.append(r)
        for key in sorted({(r['problem'],r.get('capacity',2),r['method']) for r in rr}):
            name,N,method=key
            rows=[r for r in rr if (r['problem'],r.get('capacity',2),r['method'])==key]
            valid=[r for r in rows if r.get('error') is not None]
            summary.append(dict(suite=suite,problem=name,capacity=N,method=method,runs=len(rows),exceptions=len(rows)-len(valid),
                                error_median=float(np.median([r['error'] for r in valid])),error_min=min(r['error'] for r in valid),error_max=max(r['error'] for r in valid),fits_median=float(np.median([r['fits'] for r in valid])),
                                work_limits=sum(r['status']=='work_limit' for r in rows),abnormal_returns=sum('ABNORMAL' in r['status'] for r in rows),completed_blocks=sum(r.get('completed_count',int(bool(r['completed']))) for r in rows),
                                coefficient_constraint='none'))
    output=HERE/'unconstrained-report';output.mkdir(exist_ok=True)
    with (output/'summary-all.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (output/'summary-all.json').write_text(json.dumps(summary,indent=2))
    lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Problem & $N$ & Error O & Error S & Error A & Fits O & Fits S & Fits A\\',r'\midrule']
    for name in ['forcing_two','mixed_width','narrow_mismatch']:
      for N in [4,6]:
       for suite in ['validation']:
        o=next(r for r in summary if r['suite']==suite and r['problem']==name and r['capacity']==N and r['method']=='ordinary')
        s=next(r for r in summary if r['suite']==suite and r['problem']==name and r['capacity']==N and r['method']=='stable')
        a=next(r for r in summary if r['suite']==suite and r['problem']==name and r['capacity']==N and r['method']=='adaptive')
        def number(v):
            m,e=f'{v:.3e}'.split('e');return '$'+m+r'\times10^{'+str(int(e))+'}$'
        labelname={'forcing_two':'Forcing','mixed_width':'Broad mismatch','narrow_mismatch':'Narrow mismatch'}[name]
        lines.append(f"{labelname} & {N} & {number(o['error_median'])} & {number(s['error_median'])} & {number(a['error_median'])} & {o['fits_median']:g} & {s['fits_median']:g} & {a['fits_median']:g}"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (output/'table.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
