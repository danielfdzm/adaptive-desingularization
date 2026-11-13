#!/usr/bin/env python3
"""Independent high-precision checks of unconstrained one-dimensional records."""
from pathlib import Path
import argparse, hashlib, json, math, sys
import mpmath as mp
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'core'))
from coefficient_fit import solve_unconstrained_qp
from adaptive1d import AdaptiveConfig,AdaptiveSolver1D,Problem1D,default_rho_fn
from gauss1d import AtomTable,Block,Kernel1D


def primitives(block):
    nodes=[mp.mpf(float(x)) for x in block['nodes']]
    gamma=[mp.mpf(float(x)) for x in block['gamma']]
    mult=block['mult']
    if len(nodes)==1:
        return [(nodes[0],r,(-1)**r*g) for r,g in enumerate(gamma)]
    if any(m!=1 for m in mult):
        raise ValueError('Mixed confluent/nonconfluent validation not implemented')
    weights=[mp.mpf(0)]*len(nodes)
    for r,g in enumerate(gamma):
        for j in range(r+1):
            denominator=mp.fprod(nodes[j]-nodes[k] for k in range(r+1) if k!=j)
            weights[j]+=g*mp.factorial(r)/denominator
    return [(c,0,w) for c,w in zip(nodes,weights)]


def norm2(atoms,sigma):
    def kernel(z):
        return mp.sqrt(mp.pi)*sigma*mp.exp(-z*z/(4*sigma*sigma))*(1+1/(2*sigma*sigma)-z*z/(4*sigma**4))
    return mp.fsum(a*b*(-1)**r*mp.diff(kernel,c-d,r+s)
                   for c,r,a in atoms for d,s,b in atoms)


def validate(results):
    data=json.loads((results/'shared_starts.json').read_text())
    provenance=json.loads((results/'provenance.json').read_text())
    failures=[]; errors=[]; transfers=[]
    for name,digest in provenance['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            failures.append('source_hash '+name)
    with mp.workdps(80):
        sigma=mp.mpf(float(data['meta']['sigma']))
        t=data['meta']['target']
        target=[(mp.mpf(float(c)),int(r),mp.mpf(float(a))) for c,r,a in zip(t['centers'],t['orders'],t['coeffs'])]
        target2=norm2(target,sigma)
        for run in data['runs']:
            name=f"{run['kind']}_seed{run['seed']}"
            if run.get('C') is not None or run.get('coefficient_constraint') is not None:
                failures.append('coefficient_constraint '+name)
            if run['kind']=='standard':
                atoms=[(mp.mpf(float(c)),0,mp.mpf(float(a))) for c,a in zip(run['centers'],run['weights'])]
            else:
                atoms=[atom for block in run['blocks'] for atom in primitives(block)]
            delta=atoms+[(c,r,-a) for c,r,a in target]
            e2=norm2(delta,sigma)
            if e2<0: raise ArithmeticError('negative error norm squared')
            error=float(mp.sqrt(e2/target2))
            discrepancy=abs(error-run['final_rel_err_quad'])
            errors.append(dict(run=name,relative_H1_error_80digits=error,quadrature_discrepancy=discrepancy))
            if discrepancy>max(2e-12,5e-5*error): failures.append('final_error '+name)
            for event in run.get('events',[]):
                if event['kind']!='completion': continue
                before=primitives(event['before_block']); trial=primitives(event['trial_block'])
                change=trial+[(c,r,-a) for c,r,a in before]
                actual=float(mp.sqrt(norm2(change,sigma)))
                transfers.append(dict(run=name,actual=actual,recorded_bound=event['e_val']))
                if actual>event['e_val']*(1+1e-8)+1e-13*float(mp.sqrt(target2)):
                    failures.append('transfer_bound '+name)
                if event['e_val']>event['state_tolerance'] or event['gate']>event['eps_j']:
                    failures.append('transfer_gate '+name)
    # A coefficient far beyond the old manufactured budgets must be solved without clipping.
    g,info=solve_unconstrained_qp(np.diag([1.,1e-20]),np.array([2.,3.]))
    if not np.allclose(g,[2.,3e20],rtol=1e-14) or info['constrained'] or info['rank_truncation']:
        failures.append('unconstrained_large_coefficient_solve')
    # Independently check the additive insertion line minimizer on a nonzero state.
    kern=Kernel1D(.6); target=AtomTable(np.array([0.]),np.array([2]),np.array([1.]))
    problem=Problem1D(kern,target,default_rho_fn(kern))
    solver=AdaptiveSolver1D(problem,[Block([1.],[.2])],AdaptiveConfig(N=4))
    q=float(problem.correlations(problem.residual_atoms(solver.atoms()),2,np.array([0.]))[0])
    _,insertion=solver.insertion_trial((q*q/(2*kern.kappa(2)**2),2,0.,q))
    expected=-q*q/(2*kern.kappa(2)**2)
    if abs(insertion['dJ_step']-expected)>1e-11*max(1.,abs(expected)):
        failures.append('additive_insertion_line_minimum')
    return dict(expected_runs=30,completed_runs=len(data['runs']),precision_digits=80,
                final_errors=errors,accepted_transfers=transfers,
                max_final_error_discrepancy=max(x['quadrature_discrepancy'] for x in errors),
                additive_insertion_line_checked=True,large_unconstrained_coefficient_checked=True,
                failures=failures,passed=not failures,
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);args=ap.parse_args()
    result=validate(args.results)
    (args.results/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('final_errors','accepted_transfers')},indent=2))
    assert result['passed']
