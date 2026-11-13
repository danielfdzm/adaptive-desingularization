#!/usr/bin/env python3
"""Forcing-defined 1D higher-order completion screen, no exact solution supplied.

Subclass of the archived sequential-join solver. The energy is
  .5 ||v||_H1² + lambda Q[cosh(v)-1-v²/2] - Q[f v],
with analytic whole-space Gaussian Gram matrix and fixed spatial quadrature.
The independent BVP reference is used only after optimization.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import roots_legendre, eval_hermitenorm
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SRC=ROOT/'revision-2026-11/experiments'
PB=ROOT/'adaptive-desingularization/final-experiments/pb'
sys.path.insert(0,str(PB));sys.path.insert(0,str(SRC))
import chain_solver as cs
import chain_kernels as ck
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

CASES={
    'multipole4_linear':dict(order=4,source_scale=.40,source_center=.23,amplitude=12.,reaction=0.,capacity=3),
    'multipole4_nonlinear':dict(order=4,source_scale=.40,source_center=.23,amplitude=60.,reaction=10.,capacity=3),
    'multipole5_linear':dict(order=5,source_scale=.40,source_center=-.19,amplitude=5.,reaction=0.,capacity=4),
    'multipole5_nonlinear':dict(order=5,source_scale=.40,source_center=-.19,amplitude=25.,reaction=10.,capacity=4),
}

def load(name,x):
    c=CASES[name];z=(x-c['source_center'])/c['source_scale']
    return c['amplitude']*eval_hermitenorm(c['order'],z)*np.exp(-.5*z*z)

@lru_cache(None)
def quadrature(n,L=12.):
    x,w=roots_legendre(n)
    return L*x,L*w

def rem(v):
    result=np.empty_like(v);small=np.abs(v)<.05;z=v[small]
    result[small]=z**4/24+z**6/720+z**8/40320+z**10/3628800
    result[~small]=np.cosh(v[~small])-1-.5*v[~small]**2
    return result

@lru_cache(None)
def reference(name,L=12.,tol=1e-10):
    lam=CASES[name]['reaction'];x=np.linspace(-L,L,601)
    def ode(x,y):return np.vstack((y[1],y[0]+lam*(np.sinh(y[0])-y[0])-load(name,x)))
    def bc(a,b):return np.array([a[1]-a[0],b[1]+b[0]])
    result=solve_bvp(ode,bc,x,np.zeros((2,len(x))),tol=tol,max_nodes=120000)
    if not result.success:raise RuntimeError('reference: '+result.message)
    return result

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    return v

class ForcingSolver(cs.Solver):
    def __init__(self,name,seed,method,nodes=768,maxfits=1500):
        self.problem=name;self.seed=seed
        case=CASES[name];P=case['capacity'];sigma=.55
        rng=np.random.default_rng(seed)
        starts=np.linspace(-1.5,1.5,P)+rng.uniform(-.15,.15,P)
        X,W=quadrature(nodes)
        dummy=(np.zeros((1,1)),np.ones((1,1)),np.zeros(1,dtype=int),np.zeros(1))
        super().__init__(1,sigma,dummy,[np.array([x]) for x in starts],method,X[:,None],W,
                         lam=case['reaction'],max_fits=maxfits,box=3.)
        self.initial_centers=starts.tolist();self.base_weights=W
        self.forcing=load(name,X)
        self.target2=float(W @ self.forcing**2)  # forcing normalization, never a solution norm
        self.Uq=np.zeros_like(X)

    def assemble_forcing(self,blocks,X=None,W=None,forcing=None):
        X=self.X if X is None else X
        W=self.base_weights if W is None else W
        forcing=self.forcing if forcing is None else forcing
        C,U,O,S=cs.primitives(blocks,1)
        M=S.T @ ck.h1_gram(C,U,O,C,U,O,self.sigma) @ S
        Phi=ck.point_values(X,C,U,O,self.sigma) @ S
        b=Phi.T @ torch.as_tensor(W*forcing)
        return M,b,Phi

    def evaluate(self,p,geometry):
        if self.cache is not None and np.array_equal(p,self.cache['p']):
            return self.cache['obj'],self.cache['grad']
        pt=torch.tensor(p,requires_grad=True)
        blocks=geometry.unpack(pt)
        Mt,bt,Pt=self.assemble_forcing(blocks)
        M,b,Phi=(v.detach().numpy() for v in (Mt,bt,Pt))
        g=self.fit(M,b,Phi,self.warm)
        obj,residual=self.objective_at(M,b,Phi,g)
        gt=torch.as_tensor(g)
        proxy=(.5*gt @ Mt @ gt - bt @ gt + torch.as_tensor(self.Wq*residual) @ (Pt @ gt))/self.target2
        grad=torch.autograd.grad(proxy,pt)[0].numpy()
        self.cache=dict(p=np.array(p),blocks=blocks,coef=g,obj=obj,grad=grad,M=M,b=b)
        return obj,grad

    def trial_energy(self,blocks,g):
        M,b,Phi=(v.detach().numpy() for v in self.assemble_forcing(blocks))
        return self.objective_at(M,b,Phi,g)[0],M

    def record(self):
        row=dict(fits=self.fits,iterations=self.iterations,wall=time.perf_counter()-self.t0,
                 energy=self.cache['obj'] if self.cache else None,
                 coefficient_norm=float(np.linalg.norm(self.coef)),completions=self.completions,
                 joins=self.joins,blocks=self.order_counts())
        self.trace.append(row);return row

    def fields(self,x):
        C,U,O,S=cs.primitives(self.blocks,1)
        X=torch.as_tensor(x[:,None])
        P=ck.point_values(X,C,U,O,self.sigma) @ S
        Q=(ck.point_values(X,C,U,O+1,self.sigma)*U[:,0]) @ S
        return (P @ torch.as_tensor(self.coef)).numpy(),(Q @ torch.as_tensor(self.coef)).numpy()

    def summary(self,stop):
        result=dict(problem=self.problem,seed=self.seed,method=self.method,capacity=CASES[self.problem]['capacity'],
            sigma=self.sigma,reaction=CASES[self.problem]['reaction'],initial_centers=self.initial_centers,
            fits=self.fits,iterations=self.iterations,stop=stop,
            coefficient_norm=float(np.linalg.norm(self.coef)),completions=self.completions,
            joins=self.joins,rejections=self.rejections,blocks_final=self.order_counts(),
            solve_seconds=time.perf_counter()-self.t0,trace=self.trace,events=self.events,
            final_blocks=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()} for b in self.blocks],
            final_coefficients=self.coef.tolist(),forcing_squared_norm=self.target2)
        # Save feasible optimizer endpoint before a potentially expensive reference check.
        self.pending_result=result
        sol=reference(self.problem);refined=reference(self.problem,14.,3e-11)
        checks=[]
        for n in (1024,1536):
            x,w=quadrature(n);u,ux=refined.sol(x);v,vx=self.fields(x)
            norm=float(w @ (u*u+ux*ux))
            err=math.sqrt(float(w @ ((v-u)**2+(vx-ux)**2))/norm)
            quad=.5*norm;nonlin=float(CASES[self.problem]['reaction']*w @ rem(u))
            energy=float(w @ (.5*(v*v+vx*vx)+CASES[self.problem]['reaction']*rem(v)-load(self.problem,x)*v))
            refenergy=float(w @ (.5*(u*u+ux*ux)+CASES[self.problem]['reaction']*rem(u)-load(self.problem,x)*u))
            checks.append(dict(nodes=n,error=err,energy=energy,reference_energy=refenergy,
                               nonlinear_to_quadratic_energy_ratio=nonlin/quad,reference_peak=float(np.max(np.abs(u)))))
        refdiff=sol.sol(x)-refined.sol(x)
        result.update(error=checks[-1]['error'],validation=checks,
            reference_refinement_relative_H1=math.sqrt(float(w @ np.sum(refdiff**2,axis=0))/norm),
            total_seconds=time.perf_counter()-self.t0)
        return result

def gradient_check():
    s=ForcingSolver('multipole4_nonlinear',200,'stable',nodes=768,maxfits=100)
    g=cs.Geometry(s.blocks,1,s.sigma,3.)
    x=g.x.copy();value,grad=s.evaluate(x,g);finite=[]
    for i in range(len(x)):
        h=2e-5;xp=x.copy();xm=x.copy();xp[i]+=h;xm[i]-=h
        finite.append((s.evaluate(xp,g)[0]-s.evaluate(xm,g)[0])/(2*h))
    result=dict(value=value,analytic=grad.tolist(),finite=finite,
        max_absolute_error=float(np.max(np.abs(grad-np.array(finite)))))
    if result['max_absolute_error']>2e-7:raise RuntimeError(result)
    return result

def summarize(out):
    records=[json.loads(p.read_text()) for p in (out/'runs').glob('*.json')]
    rows=[]
    for name in CASES:
      for method in ['standard_restart','stable','adaptive']:
        rr=[r for r in records if r['problem']==name and r['method']==method]
        if not rr:continue
        valid=[r for r in rr if isinstance(r.get('error'),(int,float))]
        rows.append(dict(problem=name,method=method,runs=len(rr),exceptions=len(rr)-len(valid),
            median_error=float(np.median([r['error'] for r in valid])) if valid else None,
            median_fits=float(np.median([r['fits'] for r in valid])) if valid else None,
            completions=sum(r.get('completions',0) for r in rr),
            errors=[r.get('error') for r in rr],stops=[r.get('stop') for r in rr]))
    (out/'summary.json').write_text(json.dumps(clean(rows),indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',default=list(CASES))
    ap.add_argument('--seeds',nargs='+',type=int,default=[200,201,202])
    ap.add_argument('--methods',nargs='+',default=['standard_restart','stable','adaptive'])
    ap.add_argument('--nodes',type=int,default=768);ap.add_argument('--max-fits',type=int,default=1500)
    ap.add_argument('--output',type=Path,default=HERE/'pilot');ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if any(900<=s<=909 for s in args.seeds):raise ValueError('Holdouts reserved for frozen confirmation protocol.')
    out=args.output;(out/'runs').mkdir(parents=True,exist_ok=True)
    manifest=dict(arguments={**vars(args),'output':str(out)},cases=CASES,python=platform.python_version(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__),SRC/'chain_solver.py',SRC/'chain_kernels.py',PB/'fit.py',PB/'quadrature.py']},
        starts='linspace(-1.5,1.5,P) plus U[-.15,.15] independent jitter',
        objective='exact H1 Gram + fixed quadrature nonlinear potential and load; scaled by Q[f²]',
        reference='independent BVP on [-12,12] and [-14,14]; never used in optimization',
        role='exploratory; retain all outcomes; no holdout use')
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())['source_sha256']!=manifest['source_sha256']:raise ValueError('Source changed, use fresh output.')
    else:protocol.write_text(json.dumps(manifest,indent=2))
    check=gradient_check();(out/'gradient_check.json').write_text(json.dumps(check,indent=2))
    print('gradient_check',check,flush=True)
    if args.check_only:return
    for name in args.cases:
      for seed in args.seeds:
       for method in args.methods:
        path=out/'runs'/f'{name}-{seed}-{method}.json'
        if path.exists():continue
        s=None
        try:
            s=ForcingSolver(name,seed,method,args.nodes,args.max_fits);result=s.run()
        except Exception as exc:
            result=getattr(s,'pending_result',dict(problem=name,seed=seed,method=method))
            result.update(stop='exception',exception=repr(exc))
        path.write_text(json.dumps(clean(result),indent=1));summarize(out)
        print(name,seed,method,'error',result.get('error'),'fits',result.get('fits'),
              'completed',result.get('completions'),'joins',result.get('joins'),'stop',result['stop'],flush=True)

if __name__=='__main__':main()
