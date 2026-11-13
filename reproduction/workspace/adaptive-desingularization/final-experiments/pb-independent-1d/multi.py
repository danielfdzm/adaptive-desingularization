"""Capacity-four/six extension of the independently initialized 1D PB test."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from run import (HERE,quadrature,hermite,reference,basis,rem,coefficient_fit,WorkLimit)


def problem(name):
    sigma,lam=.55,1.
    if name=='forcing_two':
        def load(x):
            return 40*(x+.72)*np.exp(-.5*((x+.72)/.32)**2)-35*(x-.78)*np.exp(-.5*((x-.78)/.36)**2)
        exact=None
    elif name in ('mixed_width','mixed_width_strong'):
        lam=10. if name.endswith('strong') else 1.
        def exact(x,k=0):
            return (.35*hermite(k,x+.72,sigma)-.38*hermite(k+1,x+.72,sigma)
                    -.3*hermite(k,x-.78,sigma)+.32*hermite(k+1,x-.78,sigma)
                    +.06*hermite(k,x-.13,1.1))
        def load(x):
            u=exact(x)
            return u-exact(x,2)+lam*(np.sinh(u)-u)
    elif name=='narrow_mismatch':
        def exact(x,k=0):
            return (.35*hermite(k,x+.72,sigma)-.38*hermite(k+1,x+.72,sigma)
                    -.3*hermite(k,x-.78,sigma)+.32*hermite(k+1,x-.78,sigma)
                    -.012*hermite(k+1,x+.72,.3)+.012*hermite(k+1,x-.78,.3))
        def load(x):
            u=exact(x)
            return u-exact(x,2)+lam*(np.sinh(u)-u)
    else:raise ValueError(name)
    return sigma,lam,load,exact


def decode(p,complete):
    blocks=[];at=0
    for done in complete:
        size=1 if done else 2
        blocks.append(p[at:at+size]);at+=size
    return blocks


def multi_basis(p,x,sigma,method,complete):
    if method=='ordinary':return basis(p,x,sigma,method)
    P=[];Q=[]
    for bb,done in zip(decode(p,complete),complete):
        pp,qq=basis(bb,x,sigma,method,done);P.append(pp);Q.append(qq)
    return torch.cat(P,1),torch.cat(Q,1)


def run_one(name,N,seed,method,n=384,maxfits=800):
    sigma,lam,load,exact=problem(name)
    X,W=quadrature(n);f=load(X);scale=float(W@(f*f))
    def kap(k):return math.sqrt(math.sqrt(math.pi)*sigma*math.factorial(2*k)/(4**k*math.factorial(k)*sigma**(2*k))*(1+(1+2*k)/(2*sigma*sigma)))
    rho=np.full(N,kap(0)) if method=='ordinary' else np.tile([kap(0),2*kap(1)],N//2)
    rng=np.random.default_rng(seed)
    centers=np.linspace(-1.5,1.5,N)+rng.uniform(-.15,.15,N)
    complete=[False]*(N//2)
    p=centers.copy() if method=='ordinary' else np.array([[centers[i:i+2].mean(),(centers[i+1]-centers[i])/2] for i in range(0,N,2)]).ravel()
    state=dict(fits=0,best=None,trace=[],events=[])
    xt=torch.from_numpy(X);wt=torch.from_numpy(W)
    start=time.monotonic()
    def evaluate(pp):
        if state['fits']>=maxfits:raise WorkLimit()
        state['fits']+=1
        pt=torch.tensor(pp,requires_grad=True)
        P,Q=multi_basis(pt,xt,sigma,method,complete)
        g,info=coefficient_fit(P.detach().numpy(),Q.detach().numpy(),W,f,lam,rho)
        gt=torch.from_numpy(g);v=P@gt;dv=Q@gt
        vv=v.detach().numpy();qq=dv.detach().numpy()
        val=float(W@(.5*(vv*vv+qq*qq)+lam*rem(vv)-f*vv))/scale
        vg=torch.from_numpy(vv+lam*(np.sinh(vv)-vv)-f)
        grad=torch.autograd.grad(torch.sum(wt*(vg*v+dv.detach()*dv)),pt)[0].numpy()/scale
        if state['best'] is None or val<state['best']['val']:
            state['best']=dict(p=np.asarray(pp).tolist(),g=g.tolist(),val=val,info=info,completed=complete.copy())
        return val,grad,g,info
    def fun(pp):
        v,j,_,_=evaluate(pp);return v,j
    flags=np.zeros(N//2,dtype=int);unchanged=0;last=np.inf;status='local_return'
    try:
        for cycle in range(60):
            bounds=[(-3.,3.)]*N if method=='ordinary' else [v for done in complete for v in ([(-3.,3.)] if done else [(-3.,3.),(1e-8,1.5)])]
            res=minimize(fun,p,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=20,ftol=1e-14,gtol=1e-11,maxls=30))
            p=res.x;val,grad,g,info=evaluate(p)
            state['trace'].append(dict(fits=state['fits'],energy=val,p=p.tolist(),gradient=float(np.max(np.abs(grad))),completed=complete.copy()))
            structural=False
            if method!='ordinary':
                blocks=decode(p,complete)
                for bi,bb in enumerate(blocks):
                    flags[bi]=flags[bi]+1 if not complete[bi] and bb[1]<=.12*sigma else 0
                    if method!='adaptive' or flags[bi]<2:continue
                    oldP,oldQ=multi_basis(torch.tensor(p),xt,sigma,method,complete)
                    nc=complete.copy();nc[bi]=True
                    nparts=[b[:1] if j==bi else b for j,b in enumerate(blocks)]
                    np_=np.concatenate(nparts)
                    newP,newQ=multi_basis(torch.tensor(np_),xt,sigma,method,nc)
                    diff=(newP-oldP).numpy()@g;diffx=(newQ-oldQ).numpy()@g
                    stateerr=math.sqrt(float(W@(diff*diff+diffx*diffx)))
                    v=oldP.numpy()@g;vx=oldQ.numpy()@g;norm=math.sqrt(float(W@(v*v+vx*vx)))
                    v=newP.numpy()@g;vx=newQ.numpy()@g
                    trial=float(W@(.5*(v*v+vx*vx)+lam*rem(v)-f*v))/scale
                    bound=bb[1]**2*(abs(g[2*bi])*kap(2)/2+abs(g[2*bi+1])*kap(3)/6)
                    allowance=1e-3*2**(-1-sum(complete))
                    accept=bound<=.01*norm and trial-val<=allowance
                    state['events'].append(dict(block=bi,fits=state['fits'],half_gap=float(bb[1]),state_change=stateerr,analytic_bound=bound,relative_bound=bound/norm,energy_change=trial-val,energy_allowance=allowance,accepted=bool(accept)))
                    if accept:
                        complete=nc;p=np_;state['best']=None;unchanged=0;last=np.inf;structural=True;break
            if structural:continue
            if val>=last-2e-14:unchanged+=1
            else:unchanged=0
            last=val
            if unchanged>=2 and not np.any(flags==1):status=str(res.message);break
        else:status='cycle_limit'
    except WorkLimit:status='work_limit'
    if state['best'] is None:raise RuntimeError('no final feasible fitted state')
    best=state['best'];p=np.array(best['p']);g=np.array(best['g']);complete=best['completed']
    checks=[]
    sol=reference(load,lam,L=12.,tol=1e-10) if exact is None else None
    for qn in [768,1024]:
        xx,ww=quadrature(qn);u,ux=sol.sol(xx) if sol is not None else (exact(xx),exact(xx,1))
        P,Q=multi_basis(torch.tensor(p),torch.from_numpy(xx),sigma,method,complete)
        v=P.numpy()@g;vx=Q.numpy()@g
        error=math.sqrt(float(ww@((v-u)**2+(vx-ux)**2))/float(ww@(u*u+ux*ux)))
        energy=float(ww@(.5*(v*v+vx*vx)+lam*rem(v)-load(xx)*v))/scale
        checks.append(dict(nodes=qn,error=error,energy=energy))
    refcheck=None
    if sol is not None:
        sb=reference(load,lam,L=14.,tol=3e-11);a=sol.sol(xx);b=sb.sol(xx)
        refcheck=math.sqrt(float(ww@np.sum((a-b)**2,axis=0))/float(ww@np.sum(b*b,axis=0)))
    return dict(problem=name,capacity=N,seed=seed,method=method,sigma=sigma,reaction=lam,initial_centers=centers.tolist(),budget=None,budget_rule=None,coefficient_constraint='none',fits=state['fits'],seconds=time.monotonic()-start,status=status,completed_count=sum(complete),error=checks[-1]['error'],energy=checks[-1]['energy'],validation=checks,reference_refinement_relative_H1=refcheck,final_p=p.tolist(),completed=complete,coefficients=g.tolist(),budget_ratio=None,coefficient_cost=best['info']['cost'],kkt=best['info']['kkt'],trace=state['trace'],events=state['events'],selection='best evaluated geometry within final representation')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--problems',nargs='+',default=['forcing_two','mixed_width','narrow_mismatch'])
    ap.add_argument('--capacities',type=int,nargs='+',default=[4,6]);ap.add_argument('--seeds',type=int,nargs='+',default=list(range(20,25)))
    ap.add_argument('--methods',nargs='+',default=['ordinary','stable','adaptive']);ap.add_argument('--nodes',type=int,default=384)
    ap.add_argument('--max-fits',type=int,default=800);ap.add_argument('--output',default='multi-results');args=ap.parse_args()
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(arguments=vars(args),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_source_sha256=hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest(),selection='best feasible evaluated geometry within final representation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[]
    for name in args.problems:
      for N in args.capacities:
       for seed in args.seeds:
        for method in args.methods:
            path=out/f'{name}-n{N}-{seed}-{method}.json'
            if path.exists():r=json.loads(path.read_text())
            else:
                try:r=run_one(name,N,seed,method,args.nodes,args.max_fits)
                except Exception as exc:r=dict(problem=name,capacity=N,seed=seed,method=method,status='exception',exception=repr(exc),error=None,energy=None,fits=None,seconds=None,completed_count=0,budget_ratio=None,kkt=None)
                path.write_text(json.dumps(r,indent=2,allow_nan=False))
            records.append(r)
            print(name,N,seed,method,'error',r['error'],'fits',r['fits'],'completed',r['completed_count'],'status',r['status'],flush=True)
    keys=['problem','capacity','seed','method','error','energy','fits','seconds','completed_count','status','budget_ratio','kkt']
    with (out/'runs.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=keys);wr.writeheader();wr.writerows([{k:r[k] for k in keys} for r in records])
    summary=[]
    for name in args.problems:
      for N in args.capacities:
       for method in args.methods:
        rr=[r for r in records if r['problem']==name and r['capacity']==N and r['method']==method];valid=[r for r in rr if r['error'] is not None]
        summary.append(dict(problem=name,capacity=N,method=method,runs=len(rr),exceptions=len(rr)-len(valid),error_median=float(np.median([r['error'] for r in valid])),energy_median=float(np.median([r['energy'] for r in valid])),fits_median=float(np.median([r['fits'] for r in valid])),completed=sum(r['completed_count'] for r in rr),work_limits=sum(r['status']=='work_limit' for r in rr)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'summary.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)

if __name__=='__main__':main()
