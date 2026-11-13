"""Independent-start 1D PB checks; all runs and pilot choices are retained.

This small benchmark uses fixed spatial quadrature and unconstrained coefficients.
Its pair-only solver is deliberately separate from the archived production
campaign. Stable and adaptive variants differ only in accepting completion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy.integrate import solve_bvp
from scipy.optimize import minimize
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pb"))


@lru_cache(maxsize=None)
def quadrature(n, L=12.0):
    x, w = np.polynomial.legendre.leggauss(n)
    return L*x, L*w


def hermite(k, y, s):
    """Spatial Gaussian derivative of order k."""
    coef = np.zeros(k+1)
    coef[k] = 1
    return (-1/s)**k*np.polynomial.hermite_e.hermeval(y/s, coef)*np.exp(-.5*(y/s)**2)


def problem(name):
    if name == "forcing_narrow":
        sigma, lam = .8, 1.0
        def load(x):
            return 45*x*np.exp(-.5*(x/.35)**2)
        exact = None
    elif name == "forcing_strong":
        sigma, lam = .8, 10.0
        def load(x):
            return 90*x*np.exp(-.5*(x/.35)**2)
        exact = None
    elif name == "off_dictionary":
        sigma, lam = .65, 1.0
        # The broad component has width different from the model's fixed width.
        def exact(x, k=0):
            return -.75*hermite(k+1,x-.23,sigma)-.025*hermite(k+1,x+.41,1.2)
        def load(x):
            u=exact(x)
            return u-exact(x,2)+lam*(np.sinh(u)-u)
    elif name == "off_dictionary_strong":
        sigma, lam = .65, 10.0
        def exact(x, k=0):
            return -.75*hermite(k+1,x-.23,sigma)-.025*hermite(k+1,x+.41,1.2)
        def load(x):
            u=exact(x)
            return u-exact(x,2)+lam*(np.sinh(u)-u)
    else:
        raise ValueError(name)
    return sigma,lam,load,exact


def reference(load, lam, L=12., tol=1e-10):
    x=np.linspace(-L,L,601)
    def fun(x,y):
        return np.vstack((y[1],y[0]+lam*(np.sinh(y[0])-y[0])-load(x)))
    # Exact linear-decay Robin conditions; nonlinear tail corrections decay cubically.
    def bc(a,b):
        return np.array([a[1]-a[0],b[1]+b[0]])
    sol=solve_bvp(fun,bc,x,np.zeros((2,len(x))),tol=tol,max_nodes=30000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def basis(p, x, sigma, method, completed=False):
    s2=sigma*sigma
    if method == "ordinary":
        y=x[:,None]-p[None,:]
        G=torch.exp(-.5*y*y/s2)
        return G,-y/s2*G
    c=p[0]
    h=torch.zeros(()) if completed else p[1]
    y=x-c
    t=y*h/s2
    near=torch.abs(t)<1e-3
    safe=torch.where(near,torch.ones_like(t),t)
    sinhc=torch.where(near,1+t*t/6+t**4/120+t**6/5040,torch.sinh(t)/safe)
    g=torch.exp(-.5*(y*y+h*h)/s2)
    E=g*torch.cosh(t)
    D=g*y/s2*sinhc
    Ex=-y/s2*E+h*h/s2*D
    Dx=-y/s2*D+E/s2
    return torch.stack((E,D),1),torch.stack((Ex,Dx),1)


def rem(v):
    # F(v)=cosh(v)-1-v²/2, with stable small-field evaluation.
    a=np.abs(v)<.05
    out=np.empty_like(v)
    z=v[a]
    out[a]=z**4/24+z**6/720+z**8/40320+z**10/3628800
    with np.errstate(over="ignore",invalid="ignore"):
        out[~a]=np.cosh(v[~a])-1-.5*v[~a]**2
    return out


def coefficient_fit(P,Q,W,f,lam,rho):
    """Damped Newton for the original energy with no coefficient constraint.

    ``rho`` is used only for the reported coefficient-cost diagnostic. A
    singular Newton system falls back to a diagonal-scaled descent direction;
    no ridge, rank truncation, projection, or coefficient clipping is used.
    """
    M=P.T@(W[:,None]*P)+Q.T@(W[:,None]*Q)
    b=P.T@(W*f)
    g=np.zeros(P.shape[1])
    scale=max(1.,np.linalg.norm(b))
    def energy(a):
        v=P@a; dv=Q@a
        return float(W@(.5*(v*v+dv*dv)+lam*rem(v)-f*v))
    val=energy(g)
    newtons=0
    for k in range(45):
        newtons+=1
        v=P@g
        with np.errstate(over="ignore",invalid="ignore"):
            residual=np.sinh(v)-v
            curvature=2*np.sinh(v/2)**2
        grad=M@g-b+lam*P.T@(W*residual)
        if np.max(np.abs(grad))<2e-13*scale:
            break
        H=M+lam*P.T@((W*curvature)[:,None]*P)
        H=.5*(H+H.T)
        diagonal=np.sqrt(np.maximum(np.diag(H),np.finfo(float).tiny))
        try:
            step=np.linalg.solve(H/diagonal[:,None]/diagonal[None,:],-grad/diagonal)/diagonal
        except np.linalg.LinAlgError:
            step=-grad/(diagonal*diagonal)
        if not np.all(np.isfinite(step)) or float(grad@step)>=0:
            step=-grad/(diagonal*diagonal)
        slope=float(grad@step)
        alpha=1.
        for _ in range(30):
            trial=g+alpha*step
            tv=energy(trial)
            if np.isfinite(tv) and tv<=val+1e-4*alpha*slope+2e-15*max(1.,abs(val)):
                break
            alpha*=.5
        else:
            break
        g,val=trial,tv
        if np.max(np.abs(alpha*step))<1e-13*max(1.,np.max(np.abs(g))):
            break
    v=P@g
    grad=M@g-b+lam*P.T@(W*(np.sinh(v)-v))
    cost=float(rho@np.abs(g))
    return g,dict(newtons=newtons,kkt=float(np.max(np.abs(grad))),gradient_inf=float(np.max(np.abs(grad))),cost=cost,multiplier=0.,coefficient_constraint='none')


class WorkLimit(Exception): pass


def run_one(name,seed,method,n=512,maxfits=700):
    sigma,lam,load,exact=problem(name)
    X,W=quadrature(n)
    f=load(X)
    scale=float(W@(f*f))
    k0=math.sqrt(math.sqrt(math.pi)*sigma*(1+1/(2*sigma*sigma)))
    k1=math.sqrt(math.sqrt(math.pi)*sigma/(2*sigma*sigma)*(1+3/(2*sigma*sigma)))
    rho=np.array([k0,k0] if method=='ordinary' else [k0,2*k1])
    rng=np.random.default_rng(seed)
    centers=np.array([-.85,.85])+rng.uniform(-.25,.25,2)
    p=centers.copy() if method=="ordinary" else np.array([centers.mean(),(centers[1]-centers[0])/2])
    state={"completed":False,"fits":0,"events":[],"trace":[],"best":None}
    xt=torch.from_numpy(X); wt=torch.from_numpy(W); ft=torch.from_numpy(f)
    t0=time.monotonic()
    def evaluate(pp, keep=True):
        if state['fits']>=maxfits: raise WorkLimit()
        state['fits']+=1
        pt=torch.tensor(pp,requires_grad=True)
        P,Q=basis(pt,xt,sigma,method,state['completed'])
        g,info=coefficient_fit(P.detach().numpy(),Q.detach().numpy(),W,f,lam,rho)
        gt=torch.from_numpy(g)
        v=P@gt; dv=Q@gt
        # Stable scalar objective value; analytical derivatives of F are used below.
        vv=v.detach().numpy(); qq=dv.detach().numpy()
        val=float(W@(.5*(vv*vv+qq*qq)+lam*rem(vv)-f*vv))/scale
        vg=torch.from_numpy(vv+lam*(np.sinh(vv)-vv)-f)
        grad=torch.autograd.grad(torch.sum(wt*(vg*v+dv.detach()*dv)),pt)[0].numpy()/scale
        if keep and (state['best'] is None or val<state['best']['val']):
            state['best']={"p":list(map(float,pp)),"g":g.tolist(),"val":val,"info":info,"completed":state['completed']}
        return val,grad,g,info
    def fun(pp):
        val,grad,_,_=evaluate(pp)
        return val,grad
    unchanged=0; flags=0; last=np.inf; status='local_return'
    try:
        for cycle in range(40):
            bounds=[(-3.,3.),(-3.,3.)] if method=='ordinary' else ([(-3.,3.)] if state['completed'] else [(-3.,3.),(1e-8,1.5)])
            res=minimize(fun,p,jac=True,method='L-BFGS-B',bounds=bounds,
                         options=dict(maxiter=20,ftol=1e-14,gtol=1e-11,maxls=30))
            p=res.x
            val,grad,g,info=evaluate(p)
            state['trace'].append(dict(fits=state['fits'],energy=val,p=p.tolist(),gradient=float(np.max(np.abs(grad))),completed=state['completed']))
            flags=flags+1 if method!='ordinary' and not state['completed'] and p[1]<=.12*sigma else 0
            if method=='adaptive' and flags>=2:
                P,Q=basis(torch.tensor(p),xt,sigma,method,False)
                PP,QQ=basis(torch.tensor(p[:1]),xt,sigma,method,True)
                dv=(PP-P).numpy()@g; dq=(QQ-Q).numpy()@g
                err=math.sqrt(float(W@(dv*dv+dq*dq)))
                norm=math.sqrt(float(W@((P.numpy()@g)**2+(Q.numpy()@g)**2)))
                v=PP.numpy()@g; dx=QQ.numpy()@g
                trial=float(W@(.5*(v*v+dx*dx)+lam*rem(v)-f*v))/scale
                # Exact analytic pair-transfer upper bound.
                def kap(k):
                    return math.sqrt(math.sqrt(math.pi)*sigma*math.factorial(2*k)/(4**k*math.factorial(k)*sigma**(2*k))*(1+(1+2*k)/(2*sigma*sigma)))
                bound=p[1]**2*(abs(g[0])*kap(2)/2+abs(g[1])*kap(3)/6)
                accept=bound<=.01*norm and trial-val<=5e-4
                state['events'].append(dict(h=float(p[1]),state_change=err,analytic_bound=bound,relative_bound=bound/norm,energy_change=trial-val,accepted=bool(accept),fits=state['fits']))
                if accept:
                    state['completed']=True;p=p[:1]
                    state['best']=None;unchanged=0;last=np.inf
                    continue
            if val>=last-2e-14: unchanged+=1
            else: unchanged=0
            last=val
            if unchanged>=2 and (flags==0 or flags>=2):
                status=str(res.message);break
        else: status='cycle_limit'
    except WorkLimit:
        status='work_limit'
    if state['best'] is None: raise RuntimeError('no valid state')
    best=state['best']; p=np.array(best['p']);g=np.array(best['g'])
    checks=[]
    for qn in [1024,1536]:
        xx,ww=quadrature(qn)
        sol=reference(load,lam,L=12.,tol=1e-10) if exact is None else None
        u,ux=sol.sol(xx) if sol is not None else (exact(xx),exact(xx,1))
        P,Q=basis(torch.tensor(p),torch.from_numpy(xx),sigma,method,best['completed'])
        v=P.numpy()@g; vx=Q.numpy()@g
        err=math.sqrt(float(ww@((v-u)**2+(vx-ux)**2))/float(ww@(u*u+ux*ux)))
        energy=float(ww@(.5*(v*v+vx*vx)+lam*rem(v)-load(xx)*v))/scale
        checks.append(dict(nodes=qn,error=err,energy=energy,norm=math.sqrt(float(ww@(u*u+ux*ux)))))
    refcheck=None
    if exact is None:
        xx,ww=quadrature(1536)
        sa=reference(load,lam,L=12.,tol=1e-10)
        sb=reference(load,lam,L=14.,tol=3e-11)
        a=sa.sol(xx);b=sb.sol(xx)
        refcheck=math.sqrt(float(ww@np.sum((a-b)**2,axis=0))/float(ww@np.sum(b*b,axis=0)))
    ordinary_norm=float(np.linalg.norm(g)) if method=='ordinary' else (None if best['completed'] else math.sqrt((g[0]**2+g[1]**2/p[1]**2)/2))
    return dict(problem=name,seed=seed,method=method,sigma=sigma,reaction=lam,capacity=2,
                initial_centers=centers.tolist(),budget=None,budget_rule=None,coefficient_constraint='none',
                fits=state['fits'],seconds=time.monotonic()-t0,status=status,completed=best['completed'],
                error=checks[-1]['error'],energy=checks[-1]['energy'],validation=checks,reference_refinement_relative_H1=refcheck,
                final_p=p.tolist(),final_coefficients=g.tolist(),ordinary_norm=ordinary_norm,
                budget_ratio=None,coefficient_cost=best['info']['cost'],kkt=best['info']['kkt'],
                events=state['events'],trace=state['trace'],selection='best feasible evaluated geometry, including line-search trials, within final structural representation')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--problems',nargs='+',default=['forcing_narrow','forcing_strong','off_dictionary','off_dictionary_strong'])
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(20,25)))
    ap.add_argument('--methods',nargs='+',default=['ordinary','stable','adaptive'])
    ap.add_argument('--nodes',type=int,default=512)
    ap.add_argument('--max-fits',type=int,default=700)
    ap.add_argument('--output',default='results')
    args=ap.parse_args()
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(args),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,domain=[-12,12],coefficient_constraint='none',notes='All cases and seeds included. No target geometry in initialization or quadrature. Returns best evaluated geometry, including line-search trials.'),indent=2))
    records=[]
    for name in args.problems:
        for seed in args.seeds:
            for method in args.methods:
                path=out/f'{name}-{seed}-{method}.json'
                if path.exists(): r=json.loads(path.read_text())
                else:
                    try:
                        r=run_one(name,seed,method,args.nodes,args.max_fits)
                    except Exception as exc:
                        r=dict(problem=name,seed=seed,method=method,status='exception',exception=repr(exc),error=None,energy=None,fits=None,seconds=None,completed=False,budget_ratio=None,kkt=None)
                    path.write_text(json.dumps(r,indent=2,allow_nan=False))
                records.append(r)
                print(name,seed,method,'error',r['error'],'fits',r['fits'],'completion',r['completed'],'h',r.get('final_p'),flush=True)
    keys=['problem','seed','method','error','energy','fits','seconds','completed','status','budget_ratio','kkt']
    with (out/'runs.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=keys);wr.writeheader()
        for r in records:wr.writerow({k:r[k] for k in keys})
    summary=[]
    for name in args.problems:
        for method in args.methods:
            rr=[r for r in records if r['problem']==name and r['method']==method]
            valid=[r for r in rr if r['error'] is not None]
            summary.append(dict(problem=name,method=method,runs=len(rr),exceptions=len(rr)-len(valid),error_median=float(np.median([r['error'] for r in valid])),energy_median=float(np.median([r['energy'] for r in valid])),fits_median=float(np.median([r['fits'] for r in valid])),completed=sum(r['completed'] for r in rr),work_limits=sum(r['status']=='work_limit' for r in rr)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'summary.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)


if __name__=='__main__':main()
