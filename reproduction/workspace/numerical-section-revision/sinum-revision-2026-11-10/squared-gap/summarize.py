#!/usr/bin/env python3
"""Pair T endpoints with all 30 frozen NC/C endpoints; compare fit counts."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'guarded-rerun-2026-11-09'/'first-order'


def main():
    rows=[]; inputs={}
    for d in (5,10,20):
        for g in (0,1):
            for s in range(10,15):
                records={}
                for label,method in [('NC','stable'),('C','adaptive'),('T','squared_gap')]:
                    directory=HERE/'results' if label=='T' else BASE/'accuracy'
                    p=directory/f'pb_d{d}_g{g}_s{s}_{method}.json'
                    records[label]=json.loads(p.read_text())
                    inputs[str(p.relative_to(HERE.parents[2]))]=hashlib.sha256(p.read_bytes()).hexdigest()
                for key in ('initial_centers','target_blocks','target_coefficients','quadrature','reaction_strength'):
                    assert records['NC'][key]==records['C'][key]==records['T'][key],key
                for key in ('relative_error','energy_gap','nonlinear_gap','coefficient_norm'):
                    assert records['NC']['trace'][0][key]==records['C']['trace'][0][key]==records['T']['trace'][0][key],key
                row=dict(dimension=d,geometry=g,seed=s)
                for label,z in records.items():
                    row[label+'_relative_error']=z['final_relative_error']
                    row[label+'_coefficient_fits']=z['fits']
                    row[label+'_iterations']=z['iterations']
                    row[label+'_stop']=z['stop']
                row['T_zero_gaps']=records['T']['zero_squared_gaps']
                row['T_serial_seconds']=records['T']['timing']['end_to_end_seconds']
                row['T_energy_gap']=records['T']['energy_gap']
                row['T_nonlinear_gap']=records['T']['nonlinear_gap']
                rows.append(row)
    with (HERE/'per-start.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for name, rr in [('all',rows)]+[(f'd{d}',[r for r in rows if r['dimension']==d]) for d in (5,10,20)]:
        z=dict(starts=len(rr),methods={})
        for label in ('NC','C','T'):
            errors=[r[label+'_relative_error'] for r in rr]
            fits=[r[label+'_coefficient_fits'] for r in rr]
            z['methods'][label]=dict(median_relative_error=st.median(errors),minimum_relative_error=min(errors),
                maximum_relative_error=max(errors),median_fits=st.median(fits),total_fits=sum(fits),
                successes_1e_6=sum(e<=1e-6 for e in errors),successes_1e_7=sum(e<=1e-7 for e in errors))
        z['median_paired_fit_ratios']={a+'_over_T':st.median(r[a+'_coefficient_fits']/r['T_coefficient_fits'] for r in rr) for a in ('NC','C')}
        z['T_error_lower_than_NC']=sum(r['T_relative_error']<r['NC_relative_error'] for r in rr)
        z['T_error_lower_than_C']=sum(r['T_relative_error']<r['C_relative_error'] for r in rr)
        z['T_median_serial_seconds']=st.median(r['T_serial_seconds'] for r in rr)
        z['T_zero_gaps_total']=sum(r['T_zero_gaps'] for r in rr)
        z['T_zero_gaps_per_start']={str(k):sum(r['T_zero_gaps']==k for r in rr) for k in range(5)}
        summaries[name]=z
    result=dict(cohort='The same 30 first-order PB starts as the guarded Table 2 NC/C cohort',
        matched_inputs_verified=True,summaries=summaries,
        accuracy='110-digit exact whole-space H1 errors; no quadrature enters the reported error',
        cost='Every coefficient fit, including initial fits, geometry trials and transfer refits',
        runtime_scope='T alone was measured in a new coordinated serial window. Full-cohort historical NC/C accuracy times were measured on a shared host and are not compared. The exact T Gram evaluator differs from the historical eight-node derivative-integral evaluator, so T timing is not a coordinate-only speed comparison.',
        historical_records='Read directly without modification from guarded-rerun-2026-11-09/first-order/accuracy',
        source_record_sha256=inputs)
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[]
    for name in ('d5','d10','d20','all'):
        z=summaries[name]; bits=[name[1:] if name!='all' else 'All']
        for label in ('NC','C','T'):
            q=z['methods'][label]
            bits.extend([f"{q['median_relative_error']:.2e}",f"{q['median_fits']:g}"])
        lines.append(' & '.join(bits)+r' \\')
    (HERE/'table-rows.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
