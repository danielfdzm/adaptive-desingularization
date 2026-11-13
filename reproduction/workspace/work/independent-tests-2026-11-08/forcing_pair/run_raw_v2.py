#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.optimize import minimize
import run as r

class CoeffFailure(Exception):pass
def raw_basis(x,b):
    z=x[None,:]-np.array(b)[:,None];B=np.exp(-z*z/2)
    return B,-z*B,(z*z-1)*B
def one(a,A,lam,seed,method,reference_dir='pilot'):
    x=np.linspace(-12,12,2401);w=r.weights(x);f=r.forcing(x,a,A);scale=max(1.,float(w@(f*f)))
    rng=np.random.default_rng(seed);beta=rng.uniform(-.8,.8);h=rng.uniform(.1,1.2)
    b=np.array([beta-h,beta+h]);B,Bx,_=raw_basis(x,b);c,_,_=r.fit(B,Bx,f,w,lam)
    p=np.r_[c,b] if method=='S_joint' else b.copy();calls=0;inner=0;best=None;trace=[];start=time.perf_counter();failures=[]
    def fun(p):
        nonlocal calls,inner,best
        calls+=1;b=p[2:] if method=='S_joint' else p
        B,Bx,Bxx=raw_basis(x,b)
        if method=='S_joint':c=p[:2];resid=None
        else:
            try:c,n,resid=r.fit(B,Bx,f,w,lam);inner+=n
            except (np.linalg.LinAlgError,ValueError,FloatingPointError) as e:
                failures.append(str(e));return 1e100,np.zeros_like(p)
        v=c@B;dv=c@Bx
        if not np.all(np.isfinite(c)) or np.max(abs(v))>100:
            failures.append('Nonfinite coefficient or |state|>100');return 1e100,np.zeros_like(p)
        q=v+lam*(np.sinh(v)-v)-f
        j=r.energy(v,dv,f,w,lam)
        gb=-c*(B@(w*q)+Bx@(w*dv))
        # center derivative uses -G' for value and -G'' for spatial derivative
        gb=-c*(Bx@(w*q)+Bxx@(w*dv))
        gc=B@(w*q)+Bx@(w*dv)
        grad=np.r_[gc,gb] if method=='S_joint' else gb
        gram=(B*w)@B.T+(Bx*w)@Bx.T
        record=dict(energy=j,weights=c.copy(),centers=b.copy(),calls=calls,coefficient_gradient=float(np.max(abs(gc))),gram_condition=float(np.linalg.cond(gram)))
        if best is None or j<best['energy']:best=record
        return j/scale,grad/scale
    def callback(p):trace.append(dict(calls=calls,best_energy=best['energy'],gram_condition=best['gram_condition']))
    bounds=[(None,None)]*2+[(-3,3)]*2 if method=='S_joint' else [(-3,3)]*2
    try:
        result=minimize(fun,p,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,options=dict(maxiter=300,maxfun=1000,gtol=1e-10,ftol=1e-14,maxls=40,maxcor=20))
        status=str(result.message);nit=int(result.nit);fun(result.x)
    except CoeffFailure as e:status='coefficient failure: '+str(e);nit=len(trace)
    elapsed=time.perf_counter()-start
    val=np.load(r.HERE/f'{reference_dir}/fine-reference-a{a}-A{A}-lambda{lam}.npz');xx=val['x'];ww=r.weights(xx)
    order=np.argsort(best['centers']);centers=best['centers'][order];weights=best['weights'][order]
    beta=float(np.mean(centers));h=float((centers[1]-centers[0])/2);stable=np.array([np.sum(weights),h*(weights[1]-weights[0])])
    # Stable validation of returned raw state separates optimization error from final cancellation.
    B,Bx,*_=r.basis(xx,beta,h*h);v=stable@B;dv=stable@Bx;u=val['u'];du=val['du'];ff=val['f'];j=r.energy(v,dv,ff,ww,lam);jstar=r.energy(u,du,ff,ww,lam)
    return dict(a=a,A=A,lam=lam,seed=seed,method=method,beta=beta,h=h,weights=weights.tolist(),centers=centers.tolist(),stable_coefficients=stable.tolist(),training_energy=best['energy'],validation_energy=j,energy_gap=j-jstar,relative_h1=float(np.sqrt(ww@((v-u)**2+(dv-du)**2)/(ww@(u*u+du*du)))),training_validation_discrepancy=j-best['energy'],ordinary_weight_norm=float(np.linalg.norm(weights)),gram_condition=best['gram_condition'],coefficient_gradient=best['coefficient_gradient'],calls=calls,inner_steps=inner,iterations=nit,seconds=elapsed,status=status,failures=failures,trace=trace)
def main():
    results=[]
    for a,A,lam in r.PROTO['pilot_cases']:
        if a>.5:continue
        for seed in r.PROTO['pilot_seeds']:
            for method in ('S_joint','S_vp'):
                row=one(a,A,lam,seed,method);results.append(row)
                (r.HERE/'raw_results.json').write_text(json.dumps(results,indent=2))
                print(a,A,lam,seed,method,'h',row['h'],'gap',row['energy_gap'],'calls',row['calls'],row['status'],flush=True)
if __name__=='__main__':main()
