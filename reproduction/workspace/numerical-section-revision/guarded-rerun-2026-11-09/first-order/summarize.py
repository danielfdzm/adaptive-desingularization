#!/usr/bin/env python3
"""Summarize the predeclared first-order cohorts without excluding failed starts."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'adaptive-desingularization/final-experiments'

def oldpath(r,method=None):
    method=method or r['method']
    if 'geometry_id' in r:
        return OLD/'pb/unconstrained/results'/f"d{r['dimension']}_g{r['geometry_id']}_seed{r['seed']}_{method}.json"
    return OLD/'linear/unconstrained-results/runs'/f"complexity_d{r['dimension']}_g{r['geometry']}_s{r['seed']}_near_Ndefault_unconstrained_{method}.json"

def group(r):
    return ('pb' if 'geometry_id' in r else 'linear_full_span', r['dimension'])

def load(mode):
    return [json.loads(p.read_text()) for p in sorted((HERE/mode).glob('*.json'))
            if p.name not in ('manifest.json','complete.json')]

def describe(rs):
    return dict(n=len(rs), error_median=statistics.median(r['final_relative_error'] for r in rs),
        error_min=min(r['final_relative_error'] for r in rs),error_max=max(r['final_relative_error'] for r in rs),
        success_1e6=sum(r['final_relative_error']<=1e-6 for r in rs),
        fits_median=statistics.median(r['fits'] for r in rs),
        completion_count=sum(r['completions'] for r in rs),
        work_limits=sum(r['stop']=='work_limit' for r in rs),
        abnormal_returns=sum(r['abnormal_returns'] for r in rs))

def main():
    rows=load('accuracy')
    timings=load('benchmark') if (HERE/'benchmark').exists() else []
    result=dict(summary_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accuracy_records=len(rows), expected_accuracy_records=72,
        complete=len(rows)==72,timing_records=len(timings),expected_timing_records=30,
        timing_complete=len(timings)==30,groups={},historical_comparison=[],
        timing_primary='end_to_end_seconds',
        timing_definition='Full measured implementation time: numerical setup + solver including detection, transfer tests, refits, restarts and original in-run diagnostic monitoring. Independent post-run 110-digit audit, imports, process startup and file I/O are outside this measure.',
        timing_alternative='end_to_end_without_diagnostic_records_seconds removes only measured record() diagnostics and is secondary.',
        directional_guard='C.4 inapplicable: these solvers have only singles and first-order pairs.',
        standard_rows='Historical S accuracy records retained; ordinary S has no new pair directions or joins and was not rerun or timed.')
    for family,d in sorted(set(group(r) for r in rows)):
        rr=[r for r in rows if group(r)==(family,d)]
        one={}
        for m in ('stable','adaptive'):
            a=[r for r in rr if r['method']==m]
            if a:one[m]=describe(a)
        starts={(r.get('geometry_id',r.get('geometry')),r['seed']):r for r in rr}
        historical=[json.loads(oldpath(r,'standard').read_text()) for r in starts.values()]
        one['standard_historical']=describe(historical)
        tt=[r for r in timings if group(r)==(family,d)]
        for m in ('stable','adaptive'):
            a=[r for r in tt if r['method']==m]
            if a:
                one[m]['serial_timing']=dict(n=len(a),
                    end_to_end_median_seconds=statistics.median(r['timing']['end_to_end_seconds'] for r in a),
                    setup_median_seconds=statistics.median(r['timing']['setup_seconds'] for r in a),
                    diagnostics_median_seconds=statistics.median(r['timing']['diagnostic_record_seconds'] for r in a),
                    numerical_end_to_end_median_seconds=statistics.median(r['timing']['end_to_end_without_diagnostic_records_seconds'] for r in a))
        pairs={}
        for r in tt:pairs.setdefault((r.get('geometry_id',r.get('geometry')),r['seed']),{})[r['method']]=r
        ratios=[p['stable']['timing']['end_to_end_seconds']/p['adaptive']['timing']['end_to_end_seconds'] for p in pairs.values() if len(p)==2]
        if ratios:one['paired_time_ratio_nc_over_c']=dict(n=len(ratios),median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios))
        secondary_ratios=[p['stable']['timing']['end_to_end_without_diagnostic_records_seconds']/p['adaptive']['timing']['end_to_end_without_diagnostic_records_seconds'] for p in pairs.values() if len(p)==2]
        if secondary_ratios:one['paired_time_ratio_without_diagnostics_nc_over_c']=dict(n=len(secondary_ratios),median=statistics.median(secondary_ratios),minimum=min(secondary_ratios),maximum=max(secondary_ratios))
        result['groups'][f'{family}_d{d}']=one
    for r in rows:
        path=oldpath(r);old=json.loads(path.read_text())
        stripped=lambda items:[{k:v for k,v in item.items() if k not in ('wall','cancellation')}
                               for item in items]
        result['historical_comparison'].append(dict(family=group(r)[0],dimension=r['dimension'],
            geometry=r.get('geometry_id',r.get('geometry')),seed=r['seed'],method=r['method'],
            error_exact=r['final_relative_error']==old['final_relative_error'],fits_exact=r['fits']==old['fits'],
            completions_exact=r['completions']==old['completions'],
            states_exact=r['blocks']==old['blocks'] and r['coefficients']==old['coefficients'],
            diagnostic_traces_exact_excluding_time=stripped(r['trace'])==stripped(old['trace']),
            events_exact_excluding_removed_cancellation=stripped(r['events'])==stripped(old['events']),
            historical_error=old['final_relative_error'],new_error=r['final_relative_error'],
            historical_path=str(path.relative_to(ROOT)),historical_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result['guard_totals']={k:sum(r['guard_counts'][k] for r in rows) for k in ('tested','eligible','unresolved','nonfinite')}
    identity=lambda r:(group(r),r.get('geometry_id',r.get('geometry')),r['seed'],r['method'])
    accuracy_index={identity(r):r for r in rows}
    result['serial_runs_reproduce_accuracy']=all(
        identity(r) in accuracy_index and all(r[k]==accuracy_index[identity(r)][k]
            for k in ('blocks','coefficients','fits','completions','final_relative_error'))
        for r in timings)
    result['independently_checked_completion_count']=sum(len(r['independent_transfer_checks']) for r in rows)
    result['all_endpoint_audits_present']=all('independent_endpoint_check_110_digits' in r for r in rows)
    result['all_transfer_state_bounds_pass']=all(t['state_bound_pass'] for r in rows for t in r['independent_transfer_checks'])
    result['all_threadpools_single_threaded']=all(p['num_threads']==1 for r in rows for p in r['reproduction']['threadpools'])
    validation_path=HERE/'pb-independent-quadrature-validation.json'
    result['pb_refined_quadrature_validation']=dict(available=validation_path.exists())
    if validation_path.exists():
        audit=json.loads(validation_path.read_text())
        result['pb_refined_quadrature_validation'].update(
            {k:audit[k] for k in ('passed','run_count','transfer_count','all_state_bounds_pass',
                                 'all_independent_energy_checks_pass','max_error_recomputation_discrepancy')})
        result['pb_refined_quadrature_validation']['sha256']=hashlib.sha256(validation_path.read_bytes()).hexdigest()
        pb_index={f"d{r['dimension']}_g{r['geometry_id']}_seed{r['seed']}_{r['method']}":r
                  for r in rows if 'geometry_id' in r}
        variations=[]
        for checked in audit['runs']:
            r=pb_index[checked['id']]
            quadratic=.5*(checked['h1_relative_error']*r['target_norm'])**2
            fine=checked['final_nonlinear_estimates'][-1]
            ref_nonlinear=fine['value'];ref_total=quadratic+ref_nonlinear
            values=[('training',r['nonlinear_gap'])]+[(e['rule'],e['value'])
                      for e in checked['final_nonlinear_estimates']]
            assert ref_nonlinear>0 and ref_total>0
            for label,value in values:
                variations.append(dict(id=checked['id'],rule=label,
                    reference_rule=fine['rule'],quadratic=quadratic,
                    reference_nonlinear=ref_nonlinear,reference_total_gap=ref_total,
                    total_gap_relative_deviation=abs(value-ref_nonlinear)/ref_total,
                    nonlinear_relative_deviation=abs(value-ref_nonlinear)/ref_nonlinear))
        result['pb_refined_quadrature_validation']['variation_summary']=dict(
            definition='For each endpoint q=0.5*(H1_error*target_H1_norm)^2; reference nonlinear term is the last power-16 rule, and reference total gap is q+that term. Compare the training nonlinear term and all five independent quadrature estimates with that reference; take absolute relative deviations and the maximum over all 60 NC/C endpoints.',
            compared_estimates=len(variations),
            total_gap_max=max(variations,key=lambda r:r['total_gap_relative_deviation']),
            nonlinear_max=max(variations,key=lambda r:r['nonlinear_relative_deviation']),
            source_input_sha256=audit['new_input_sha256'],
            source_hashes_still_match=all(hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h
                                         for p,h in audit['new_input_sha256'].items()))
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    flat=[]
    for r in rows:
        flat.append(dict(family=group(r)[0],dimension=r['dimension'],geometry=r.get('geometry_id',r.get('geometry')),
                         seed=r['seed'],method=r['method'],error=r['final_relative_error'],fits=r['fits'],
                         completions=r['completions'],stop=r['stop'],**r['guard_counts']))
    if flat:
        with (HERE/'accuracy.csv').open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    time_rows=[]
    for name,g in result['groups'].items():
        if not all('serial_timing' in g.get(m,{}) for m in ('stable','adaptive')):continue
        a,b=g['stable']['serial_timing'],g['adaptive']['serial_timing']
        time_rows.append(dict(study=name,paired_starts=a['n'],
            nc_end_to_end_seconds=a['end_to_end_median_seconds'],
            c_end_to_end_seconds=b['end_to_end_median_seconds'],
            nc_without_diagnostics_seconds=a['numerical_end_to_end_median_seconds'],
            c_without_diagnostics_seconds=b['numerical_end_to_end_median_seconds'],
            median_paired_nc_over_c=g['paired_time_ratio_nc_over_c']['median']))
    if time_rows:
        with (HERE/'serial-runtime.csv').open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(time_rows[0]));w.writeheader();w.writerows(time_rows)
    print(json.dumps({k:v for k,v in result.items() if k!='historical_comparison'},indent=2))

if __name__=='__main__':main()
