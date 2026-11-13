#!/usr/bin/env python3
"""Run one frozen supplementary control; shared-host times are not benchmarks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, importlib.util, json, math, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), args.output
    task = next(t for t in json.loads((HERE/'tasks.json').read_text()) if t['id'] == args.id)
    root = HERE/'workspace'
    basepath = root/'adaptive-desingularization/final-experiments/pb-independent-2d'
    if task['snapshot']:
        module = load('forcing_control', root/'work/independent-tests-2026-11-08/forcing_2d'/
                      (task['snapshot'].replace('-','_')+'.py'))
        ind = module.ind
        build = lambda: module.Solver(task['case'],task['seed'],task['method'],task['max_fits'],task['order'])
    elif task['energy_factor'] is not None:
        module = load('allowance_control', basepath/'sensitivity.py')
        ind = module.m
        build = lambda: module.StrictGate(task['case'],task['seed'],task['method'],task['max_fits'],task['order'],
                                          task['perturbation'],4,'lattice',energy_factor=task['energy_factor'])
    else:
        ind = load('independent_control', basepath/'run.py')
        build = lambda: ind.IndependentSolver(task['case'],task['seed'],task['method'],task['max_fits'],
                                              task['order'],task['perturbation'],4,'lattice')
    import numpy as np
    import scipy, torch, mpmath as mp
    from threadpoolctl import threadpool_info
    from kernels import _mp_primitives, _mp_inner, high_precision_error
    from quadrature import weighted_bregman
    started = time.perf_counter()
    solver = build()
    setup_seconds = time.perf_counter()-started
    solver.guard_counts = dict(tested=0,eligible=0,unresolved=0,nonfinite=0)
    result = solver.run()
    elapsed = time.perf_counter()-started
    if task['cohort']=='recovery': result['variant']=task['method']
    if task['energy_factor'] is not None: result['energy_factor']=task['energy_factor']
    audit_started = time.perf_counter()
    if task['snapshot']:
        x,u,ux,uy,meta = module.reference_cached(task['case'],module.CASES[task['case']],n=512,L=10.)
        X,Y=np.meshgrid(x,x,indexing='ij'); pts=np.column_stack((X.ravel(),Y.ravel()))
        v,g=module.values_grad(result['blocks'],np.asarray(result['coefficients']),pts)
        err2=float(np.sum((v-u.ravel())**2+(g[:,0]-ux.ravel())**2+(g[:,1]-uy.ravel())**2)*(x[1]-x[0])**2)
        endpoint=dict(kind='direct refined Fourier grid',relative_H1=math.sqrt(err2/meta['norm2']),
                      reference=meta,training_adequacy='underresolved compact-source pilot' if solver.cfg['shape']=='compact' else 'exploratory smooth-source pilot')
        old_checks=json.loads((HERE/'provenance/forcing-historical-endpoint-validation.json').read_text())
        old=next(r for r in old_checks['endpoints'] if r['file']==task['snapshot']+'/'+Path(task['historical']).name)
        endpoint['historical_relative_H1_direct_refined']=old['relative_H1_direct_refined']
        endpoint['historical_direct_error_exactly_reproduced']=endpoint['relative_H1']==old['relative_H1_direct_refined']
    else:
        with mp.workdps(110):
            A=[(*p,mp.mpf(.45)) for p in _mp_primitives(solver.blocks,solver.coef)]
            T=ind.target_primitives(solver.terms)
            norm2=ind.inner(T,T); err2=ind.inner(A,A)-2*ind.inner(A,T)+norm2
            assert err2>=0
            rel=float(mp.sqrt(err2/norm2))
        endpoint=dict(kind='analytic mixed-width Gaussian H1, 110 decimal digits',relative_H1=rel,
                      discrepancy_from_recorded=abs(rel-result['final_relative_error']))
        assert endpoint['discrepancy_from_recorded']<=max(1e-20,1e-10*rel)
    transfer_checks=[]
    VX,VW=ind.rule(120,5.)
    force=module.force(VX,solver.cfg) if task['snapshot'] else ind.field(VX,solver.terms)[0]-ind.field(VX,solver.terms)[2]+np.sinh(ind.field(VX,solver.terms)[0])-ind.field(VX,solver.terms)[0]
    for e in result['events']:
        if e['kind']!='completion': continue
        i=next(i for i,b in enumerate(e['before_blocks']) if b['id']==e['block_id'])
        ix=ind.base.offsets(e['before_blocks']); coef=e['before_coefficients'][ix[i]:ix[i+1]]
        B=[e['before_blocks'][i]]; A=[e['trial_blocks'][i]]
        actual=high_precision_error(A,coef,B,coef,dps=110)['norm']
        bound_pass=actual<=e['state_error_bound']*(1+1e-8)+1e-14*math.sqrt(solver.target2)
        tolerance_pass=actual<=e['state_tolerance']*(1+1e-8)
        assert bound_pass and tolerance_pass
        with mp.workdps(110):
            negate=lambda ps:[(c,[-v for v in vs],-a) for c,vs,a in ps]
            delta=_mp_primitives(A,coef)+negate(_mp_primitives(B,coef))
            state=_mp_primitives(e['before_blocks'],e['before_coefficients'])
            quadratic=float(_mp_inner(state,delta,.45)+_mp_inner(delta,delta,.45)/2)
        before=ind.column_values(ind.base.arrays(e['before_blocks']),torch.as_tensor(VX)).numpy()@np.asarray(e['before_coefficients'])
        trial=ind.column_values(ind.base.arrays(e['trial_blocks']),torch.as_tensor(VX)).numpy()@np.asarray(e['trial_coefficients'])
        nonlinear=weighted_bregman(trial,np.zeros_like(trial),solver.reaction_strength*VW)-weighted_bregman(before,np.zeros_like(before),solver.reaction_strength*VW)
        refined_gate=float(quadratic-VW@(force*(trial-before))+nonlinear)
        # Quadrature is an independent empirical refinement check, not a continuum certificate.
        tolerance=1e-10*solver.target2
        transfer_checks.append(dict(fits=e['fits'],block_id=e['block_id'],state_change_norm_110_digits=actual,
            state_bound_pass=bound_pass,state_tolerance_pass=tolerance_pass,recorded_gate=e['gate'],allowance=e['eps_j'],
            refined_gate_order120_halfwidth5=refined_gate,quadrature_numerical_tolerance=tolerance,
            refined_gate_within_allowance_and_numerical_tolerance=refined_gate<=e['eps_j']+tolerance))
    result['guard_counts']=solver.guard_counts
    result['directional_guard']=dict(applicable=False,reason='Only singles and first-order pairs; no directional joins or higher-order charts.')
    result['supplementary_task']=task
    result['independent_endpoint_check']=endpoint
    result['independent_transfer_checks']=transfer_checks
    result['timing']=dict(context='accuracy-only shared host; not controlled runtime evidence',setup_seconds=setup_seconds,
        end_to_end_seconds=elapsed,independent_postrun_audit_seconds=time.perf_counter()-audit_started,
        definition='perf_counter from construction through run return; includes original monitoring and post-run energy diagnostics inside run; excludes added independent audits, imports and I/O')
    sources=list((HERE/'workspace').rglob('*.py'))+[HERE/'pair_guard.py',Path(__file__)]
    result['reproduction']=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        torch=torch.__version__,mpmath=mp.__version__,threadpools=threadpool_info(),torch_threads=torch.get_num_threads(),
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        tasks_sha256=hashlib.sha256((HERE/'tasks.json').read_bytes()).hexdigest(),pid=os.getpid())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(ind.base.plain(result),indent=2)+'\n')
    print(json.dumps(dict(id=task['id'],error=result['final_relative_error'],fits=result['fits'],seconds=elapsed,guard=solver.guard_counts)),flush=True)


if __name__=='__main__': main()
