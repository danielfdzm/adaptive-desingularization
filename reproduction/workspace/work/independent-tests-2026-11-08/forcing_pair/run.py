#!/usr/bin/env python3
"""Prescribed-forcing PDE experiment; independent of manuscript implementations."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp, simpson
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
PROTO=json.loads((HERE/'protocol.json').read_text())
def weights(x):
    w=np.ones(len(x)); w[1:-1:2]=4; w[2:-1:2]=2
    return w*(x[1]-x[0])/3
def forcing(x,a,A): return A*x/a**2*np.exp(-x*x/(2*a*a))
def basis(x,beta,t):
    z=x-beta; h=np.sqrt(max(t,0.)); g=np.exp(-z*z/2)
    if t<1e-4:
        E=np.zeros_like(x); D=np.zeros_like(x); Et=np.zeros_like(x); Dt=np.zeros_like(x)
        import math
        for k in range(7):
            E+=z**(2*k)*t**k/math.factorial(2*k)
            D+=z**(2*k+1)*t**k/math.factorial(2*k+1)
            if k:
                Et+=k*z**(2*k)*t**(k-1)/math.factorial(2*k)
                Dt+=k*z**(2*k+1)*t**(k-1)/math.factorial(2*k+1)
        g=g*np.exp(-t/2); Et=g*(Et-E/2); Dt=g*(Dt-D/2); E=g*E; D=g*D
    else:
        gp=np.exp(-(z-h)**2/2); gm=np.exp(-(z+h)**2/2)
        E=(gp+gm)/2; D=(gp-gm)/(2*h)
        Et=(-E+z*D)/2; Dt=-D/2+(z*E-D)/(2*t)
    Ex=-z*E+t*D; Dx=-z*D+E
    Exx=(z*z-1+t)*E-2*z*t*D; Dxx=(z*z-1+t)*D-2*z*E
    Ext=-z*Et+D+t*Dt; Dxt=-z*Dt+Et
    return np.array([E,D]),np.array([Ex,Dx]),np.array([-Ex,-Dx]),np.array([-Exx,-Dxx]),np.array([Et,Dt]),np.array([Ext,Dxt])
def energy(v,dv,f,w,lam):
    return float(w@(.5*(v*v+dv*dv)+lam*(np.cosh(v)-1-.5*v*v)-f*v))
def fit(B,Bx,f,w,lam):
    gram=(B*w)@B.T+(Bx*w)@Bx.T; load=B@(w*f)
    c=np.linalg.solve(gram,load)
    if not lam: return c,1,0.
    c*=min(1.,3/max(np.max(np.abs(c@B)),1e-30))
    for k in range(80):
        v=c@B; dv=c@Bx
        grad=B@(w*(v+lam*(np.sinh(v)-v)-f))+Bx@(w*dv)
        if np.linalg.norm(grad,np.inf)<1e-10*(1+np.linalg.norm(load,np.inf)): break
        H=(B*(w*(1+lam*(np.cosh(v)-1))))@B.T+(Bx*w)@Bx.T
        step=np.linalg.solve(H,grad); old=energy(v,dv,f,w,lam); alpha=1.
        for _ in range(40):
            trial=c-alpha*step; tv=trial@B
            if np.max(np.abs(tv))<100 and energy(tv,trial@Bx,f,w,lam)<=old-1e-4*alpha*(grad@step)+1e-13*(1+abs(old)):
                c=trial; break
            alpha*=.5
        else: break
    return c,k+1,float(np.linalg.norm(grad,np.inf))
def reference(a,A,lam,L=14,tol=1e-9):
    xx=np.linspace(-L,L,701)
    def ode(x,y): return np.vstack((y[1],y[0]+lam*(np.sinh(y[0])-y[0])-forcing(x,a,A)))
    def bc(ya,yb): return np.array([ya[0],yb[0]])
    r=solve_bvp(ode,bc,xx,np.zeros((2,len(xx))),tol=tol,max_nodes=40000)
    if not r.success: raise RuntimeError(r.message)
    return r

class Complete(Exception):
    def __init__(self,p): self.p=np.array(p)
def run_one(a,A,lam,seed,method):
    x=np.linspace(-12,12,2401); w=weights(x); f=forcing(x,a,A)
    scale=max(1.,float(w@(f*f))); calls=0; inner=0; iters=0; flags=0; events=[]; trace=[]
    rng=np.random.default_rng(seed); beta0=rng.uniform(-.8,.8); h0=rng.uniform(.1,1.2)
    start=time.perf_counter(); last={}; completed=False
    def evaluate(p,chart):
        nonlocal calls,inner,last
        beta=float(p[0]); t=0. if chart=='boundary' else float(p[1])**(2 if chart=='h' else 1)
        B,Bx,Bb,Bxb,Bt,Bxt=basis(x,beta,t); c,n,g=fit(B,Bx,f,w,lam); calls+=1; inner+=n
        v=c@B; dv=c@Bx; j=energy(v,dv,f,w,lam)
        residual=v+lam*(np.sinh(v)-v)-f
        gb=float(w@(residual*(c@Bb)+dv*(c@Bxb))); gt=float(w@(residual*(c@Bt)+dv*(c@Bxt)))
        grad=[gb] if chart=='boundary' else [gb,gt*(2*float(p[1]) if chart=='h' else 1.)]
        last={'beta':beta,'t':t,'c':c,'energy':j,'coeff_grad':g,'state':v,'dstate':dv}
        return j/scale,np.array(grad)/scale
    def scan(p,trigger):
        nonlocal flags
        evaluate(p,'h'); h=float(p[1]); norm=np.sqrt(w@(last['state']**2+last['dstate']**2))
        rho=np.exp(-h*h)*(1.5-h*h)/1.5; kappa=np.sqrt(max(0.,1-abs(rho)))
        flag=2*h<=.5 and kappa<=.1; flags=flags+1 if flag else 0
        if flags<2:return
        B,Bx,*_=basis(x,p[0],0.); v=last['c']@B; dv=last['c']@Bx
        change=np.sqrt(w@((v-last['state'])**2+(dv-last['dstate'])**2)); delta=energy(v,dv,f,w,lam)-last['energy']
        accept=change<=.01*norm and delta<=.0005*scale
        events.append(dict(trigger=trigger,h=h,kappa=kappa,change=change,state_tolerance=.01*norm,energy_change=delta,allowance=.0005*scale,accepted=bool(accept)))
        if accept: raise Complete(p)
    def callback(p):
        nonlocal iters
        iters+=1; trace.append({'iteration':iters,'calls':calls,'energy':last['energy'],'beta':last['beta'],'h':np.sqrt(last['t'])})
        if method=='C' and iters%5==0:scan(p,'periodic')
    chart='t' if method=='SC' else 'h'; p=np.array([beta0,h0*h0 if chart=='t' else h0]); returns=[]
    opt=dict(maxiter=300,maxfun=1000,gtol=1e-10,ftol=1e-14,maxls=40,maxcor=20)
    try:
        for cycle in range(2 if method=='C' else 1):
            r=minimize(lambda p:evaluate(p,chart),p,jac=True,method='L-BFGS-B',bounds=[(-3,3),(0,9) if chart=='t' else (1e-6,3)],callback=callback,options=opt)
            p=r.x; returns.append(str(r.message))
            if method=='C':scan(p,'local_return')
            if method!='C' or flags==0:break
    except Complete as done:
        completed=True; chart='boundary'; p=done.p[:1]
        r=minimize(lambda p:evaluate(p,'boundary'),p,jac=True,method='L-BFGS-B',bounds=[(-3,3)],options=opt)
        p=r.x; returns.append(str(r.message))
    evaluate(p,chart); elapsed=time.perf_counter()-start
    return dict(a=a,A=A,lam=lam,seed=seed,method=method,beta0=beta0,h0=h0,beta=last['beta'],h=float(np.sqrt(last['t'])),coefficients=last['c'].tolist(),training_energy=last['energy'],coeff_grad=last['coeff_grad'],completed=completed,calls=calls,inner_steps=inner,iterations=iters,seconds=elapsed,returns=returns,events=events,trace=trace)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',default='pilot'); args=parser.parse_args()
    out=HERE/args.phase; out.mkdir(exist_ok=True)
    rows=[]; refs=[]; seeds=PROTO['pilot_seeds'] if args.phase=='pilot' else PROTO['reserved_seeds']
    cases=PROTO['pilot_cases']
    for a,A,lam in cases:
        ref=reference(a,A,lam); valid=reference(a,A,lam,16,1e-10)
        x=np.linspace(-16,16,6401); w=weights(x); u,du=valid.sol(x); f=forcing(x,a,A)
        norm=float(w@(u*u+du*du)); jstar=energy(u,du,f,w,lam)
        xi=np.linspace(-14,14,5601); wi=weights(xi); u1,d1=ref.sol(xi); u2,d2=valid.sol(xi)
        err=np.sqrt(wi@((u1-u2)**2+(d1-d2)**2)/norm)
        rec=dict(a=a,A=A,lam=lam,reference_h1_norm=np.sqrt(norm),max_u=float(np.max(abs(u))),reference_energy=jstar,reference_domain_tolerance_relative_h1=float(err),bvp_nodes=valid.x.size,bvp_residual=float(np.max(valid.rms_residuals)))
        refs.append(rec); np.savez(out/f'reference-a{a}-A{A}-lambda{lam}.npz',x=x,u=u,du=du,f=f)
        for seed in seeds:
            for method in ('NC','C','SC'):
                row=run_one(a,A,lam,seed,method); B,Bx,*_=basis(x,row['beta'],row['h']**2); c=np.array(row['coefficients']); v=c@B; dv=c@Bx
                row['relative_h1']=float(np.sqrt(w@((v-u)**2+(dv-du)**2)/norm)); row['validation_energy']=energy(v,dv,f,w,lam); row['energy_gap']=row['validation_energy']-jstar
                row['quadrature_energy_discrepancy']=row['validation_energy']-row['training_energy']
                row['ordinary_weight_norm']=float(np.sqrt((c[0]**2+(c[1]/row['h'])**2)/2)) if row['h'] else None
                row['stable_coefficient_norm']=float(np.linalg.norm(c)); rows.append(row)
                (out/'results.json').write_text(json.dumps(rows,indent=2)); (out/'references.json').write_text(json.dumps(refs,indent=2))
                print(a,A,lam,seed,method,'h',f"{row['h']:.3g}",'relH1',f"{row['relative_h1']:.6g}",'gap',f"{row['energy_gap']:.6g}",'calls',row['calls'],flush=True)
    summary=[]
    for a,A,lam in cases:
        for method in ('NC','C','SC'):
            rr=[r for r in rows if (r['a'],r['A'],r['lam'],r['method'])==(a,A,lam,method)]
            summary.append(dict(a=a,A=A,lam=lam,method=method,median_relative_h1=float(np.median([r['relative_h1'] for r in rr])),median_gap=float(np.median([r['energy_gap'] for r in rr])),median_calls=float(np.median([r['calls'] for r in rr])),completions=sum(r['completed'] for r in rr),boundary=sum(r['h']<1e-5 for r in rr)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
