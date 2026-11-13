#!/usr/bin/env python3
"""Generate flat measured outcomes and compact summaries from completed runs."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def source_name(path):
    repository=Path(__file__).resolve().parents[2]
    try:return str(path.resolve().relative_to(repository))
    except ValueError:return str(path.resolve())

def build(directories,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for directory in directories:
        directory=Path(directory)
        for path in sorted(directory.glob('d*_g*_seed*_*.json')):
            r=json.loads(path.read_text());fit=r.get('coefficient_fit') or {}
            rows.append(dict(study=directory.name,source=source_name(path),dimension=r['dimension'],
                geometry_id=r['geometry_id'],seed=r['seed'],method=r['method'],reaction_strength=r['reaction_strength'],
                power=r['quadrature']['power'],quadrature_seed=r['quadrature']['seed'],nodes=r['quadrature']['nodes'],
                relative_error=r['final_relative_error'],best_relative_error=r['best_relative_error'],
                initial_fitted_relative_error=r['trace'][0]['relative_error'],
                normalized_energy_gap=r['energy_gap']/r['target_norm']**2,
                nonlinear_gap=r['nonlinear_gap'],chi_nl_training=r['chi_nl_training'],
                target_sampled_peak_lower_bound=r['target_sampled_peak'],
                fits=r['fits'],wall_seconds=r['wall'],setup_seconds=r['setup_wall'],
                worker_peak_rss_bytes=r['worker_peak_rss_bytes'],
                nonlinear_value_calls=r['nonlinear_value_calls'],nonlinear_derivative_calls=r['nonlinear_derivative_calls'],
                quadrature_point_evaluations=r['quadrature_point_evaluations'],
                completions=r['completions'],stop=r['stop'],abnormal_returns=r['abnormal_returns'],
                projected_gradient=r['projected_gradient_max_abs'],coefficient_kkt_scaled=fit.get('kkt_scaled_residual'),
                coefficient_norm=r['coefficient_norm'],budget_cost_ratio=None if r['C'] is None else r['cost']/r['C'],
                max_gram_condition=r['max_cond'],
                final_normalized_gram_condition=r['trace'][-1]['stored_gram_condition'],
                final_min_positive_halfgap=r['trace'][-1]['min_positive_halfgap'],
                **{f'time_to_{t}':r['time_to_error'].get(t) for t in ('0.001','0.0001','1e-05','1e-06','1e-07')}))
    if not rows:raise ValueError('No run files')
    with (output/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'runs.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    groups=defaultdict(list)
    for r in rows:groups[(r['study'],r['dimension'],r['reaction_strength'],r['method'])].append(r)
    summary=[]
    for (study,d,reaction,method),rs in sorted(groups.items()):
        errors=np.array([r['relative_error'] for r in rs])
        summary.append(dict(study=study,dimension=d,reaction_strength=reaction,method=method,runs=len(rs),
            initial_fitted_error_median=float(np.median([r['initial_fitted_relative_error'] for r in rs])),
            error_median=float(np.median(errors)),error_q25=float(np.quantile(errors,.25)),
            error_q75=float(np.quantile(errors,.75)),error_min=float(errors.min()),error_max=float(errors.max()),
            success_1e4=int(np.sum(errors<=1e-4)),success_1e6=int(np.sum(errors<=1e-6)),success_1e7=int(np.sum(errors<=1e-7)),
            wall_median=float(np.median([r['wall_seconds'] for r in rs])),
            fits_median=float(np.median([r['fits'] for r in rs])),
            completions_total=sum(r['completions'] for r in rs),
            work_limits=sum(r['stop']=='work_limit' for r in rs),
            runs_with_abnormal_returns=sum(r['abnormal_returns']>0 for r in rs),
            max_projected_gradient=max((r['projected_gradient'] for r in rs if r['projected_gradient'] is not None),default=None),
            max_coefficient_kkt_scaled=max((r['coefficient_kkt_scaled'] for r in rs if r['coefficient_kkt_scaled'] is not None),default=None),
            missing_stationarity_diagnostics=sum(r['projected_gradient'] is None or r['coefficient_kkt_scaled'] is None for r in rs)))
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    primary=[r for r in summary if r['study'] in ('results','calibrated')]
    if primary:
        with (output/'primary-summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(primary[0]));writer.writeheader();writer.writerows(primary)
        (output/'primary-summary.json').write_text(json.dumps(primary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'report')
    args=parser.parse_args();build(args.directories,args.output)
