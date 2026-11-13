#!/usr/bin/env python3
"""Independent endpoint/transfer validation and nonlinear-strength estimates.

Validation uses both the original Gaussian proposal covariance sigma^2/2
and the narrower covariance sigma^2/4. The latter resolves quartic localized
nonlinearities more efficiently in large d. Neither rule is a certificate.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from kernels import column_values, high_precision_error, validation_checks
from quadrature import make_quadrature, weighted_bregman
from config import SIGMA

def values(blocks,coefficients,x,chunk=16384):
    out=np.empty(len(x))
    with torch.no_grad():
        for first in range(0,len(x),chunk):
            out[first:first+chunk]=(column_values(blocks,x[first:first+chunk])@torch.as_tensor(coefficients)).numpy()
    return out

def identity(run):
    return f"d{run['dimension']}_g{run['geometry_id']}_seed{run['seed']}_{run['method']}"

def validate(directory,coarse=14,fine=16):
    directory=Path(directory)
    runs=[json.loads(p.read_text()) for p in sorted(directory.glob('d*_g*_seed*_*.json'))]
    if not runs:
        raise ValueError('no per-run results')
    start=time.perf_counter()
    provenance=json.loads((directory/'provenance.json').read_text())
    root=Path(__file__).resolve().parent.parent
    source_checks={name:hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
                   for name,sha in provenance['source_hashes'].items()}
    expected=(len(provenance['dimensions'])*len(provenance['geometries'])*
              (provenance['seeds'][1]-provenance['seeds'][0])*len(provenance['methods']))
    validation={'rigorous_quadrature_certificate':False,'mpmath_decimal_precision':80,
                'run_count':len(runs),'expected_run_count':expected,
                'complete':len(runs)==expected,'source_checks':source_checks,
                'kernel_checks':validation_checks(),'runs':[],'groups':[]}
    records={}
    for run in runs:
        tb,tc=run['target_blocks'],run['target_coefficients']
        exact=high_precision_error(run['blocks'],run['coefficients'],tb,tc)
        r=dict(id=identity(run),dimension=run['dimension'],geometry_id=run['geometry_id'],
               method=run['method'],seed=run['seed'],
               h1_relative_error=exact['relative'],
               recorded_error_discrepancy=abs(exact['relative']-run['final_relative_error']),
               final_nonlinear_estimates=[],transfers=[])
        for event in run['events']:
            if event['kind']!='completion':continue
            before=high_precision_error(event['before_blocks'],event['before_coefficients'],tb,tc)['norm']
            trial=high_precision_error(event['trial_blocks'],event['trial_coefficients'],tb,tc)['norm']
            change=high_precision_error(event['trial_blocks'],event['trial_coefficients'],event['before_blocks'],event['before_coefficients'])['norm']
            r['transfers'].append(dict(block_id=event['block_id'],fits=event['fits'],
                    analytic_quadratic_change=.5*(trial*trial-before*before),
                    state_change_norm=change,state_error_bound=event['state_error_bound'],
                    state_bound_pass=change<=event['state_error_bound']*(1+1e-6)+1e-13,
                    allowance=event['eps_j'],training_increment=event['gate'],independent_increments=[]))
        records[identity(run)]=r
    groups=sorted({(r['dimension'],r['geometry_id'],r['reaction_strength']) for r in runs})
    for d,g,reaction in groups:
        selected=[r for r in runs if (r['dimension'],r['geometry_id'],r['reaction_strength'])==(d,g,reaction)]
        tb,tc=selected[0]['target_blocks'],selected[0]['target_coefficients']
        centers=np.array([b['c'] for b in tb]);norm2=selected[0]['target_norm']**2
        estimates=[]
        # Three independent scrambles at the narrower proposal, one broader
        # diagnostic and one refinement. All are independent of training.
        specs=[(coarse,9101,SIGMA),(coarse,9201,SIGMA/math.sqrt(2)),
               (coarse,9202,SIGMA/math.sqrt(2)),(coarse,9203,SIGMA/math.sqrt(2)),
               (fine,9201,SIGMA/math.sqrt(2))]
        for power,seed,proposal_sigma in specs:
            x,w,meta=make_quadrature(centers,sigma=proposal_sigma,power=power,seed=seed)
            w*=reaction
            u=values(tb,tc,x)
            label=f"p{power}_s{seed}_proposal{proposal_sigma:.12g}"
            chi=weighted_bregman(u,np.zeros_like(u),w)/(.5*norm2)
            estimates.append(dict(rule=label,chi_nl=chi,sampled_peak_lower_bound=float(max(abs(u))),
                                  power=power,seed=seed,points=len(x),proposal_covariance=proposal_sigma**2/2))
            for run in selected:
                record=records[identity(run)]
                v=values(run['blocks'],run['coefficients'],x)
                nonlinear=weighted_bregman(v,u,w)
                record['final_nonlinear_estimates'].append(dict(rule=label,value=nonlinear))
                completed=[e for e in run['events'] if e['kind']=='completion']
                for event,tr in zip(completed,record['transfers']):
                    before=values(event['before_blocks'],event['before_coefficients'],x)
                    trial=values(event['trial_blocks'],event['trial_coefficients'],x)
                    increment=(tr['analytic_quadratic_change']+weighted_bregman(trial,u,w)-weighted_bregman(before,u,w))
                    tr['independent_increments'].append(dict(rule=label,value=increment,
                                  allowance_margin=tr['allowance']-increment,
                                  passes=bool(increment<=tr['allowance']+1e-12*norm2)))
            print(f'validated d={d} geometry={g} lambda={reaction:g} {label} chi={chi:.5g}',flush=True)
        narrow=[e['chi_nl'] for e in estimates[1:4]]
        validation['groups'].append(dict(dimension=d,geometry_id=g,reaction_strength=reaction,
                 estimates=estimates,chi_narrow_coarse_mean=float(np.mean(narrow)),
                 chi_narrow_coarse_std=float(np.std(narrow,ddof=1)),
                 chi_narrow_refined=estimates[-1]['chi_nl']))
    validation['runs']=list(records.values())
    transfers=[t for r in records.values() for t in r['transfers']]
    validation.update(transfer_count=len(transfers),
        all_state_bounds_pass=all(t['state_bound_pass'] for t in transfers),
        all_independent_energy_checks_pass=all(q['passes'] for t in transfers for q in t['independent_increments']),
        max_error_recomputation_discrepancy=max(r['recorded_error_discrepancy'] for r in records.values()),
        elapsed_seconds=time.perf_counter()-start)
    validation['passed']=(validation['complete'] and all(source_checks.values()) and
        validation['all_state_bounds_pass'] and validation['all_independent_energy_checks_pass'] and
        validation['max_error_recomputation_discrepancy']<1e-12)
    return validation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--coarse',type=int,default=14)
    parser.add_argument('--fine',type=int,default=16)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=validate(args.results,args.coarse,args.fine)
    path=args.output or args.results/'validation.json'
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('runs','groups','kernel_checks')},indent=2))
