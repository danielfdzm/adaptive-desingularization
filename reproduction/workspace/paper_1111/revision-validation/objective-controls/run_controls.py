#!/usr/bin/env python3
"""Local pair controls: objective cancellation, common stopping, positive gap floors.

All methods refit the two stable coefficients at every trial geometry.
The residual objective is evaluated directly on a whole-line truncation;
the shifted Gram objective deliberately retains the cancellation-prone
evaluation used in the historical comparisons. Endpoint errors and the
canonical (beta,t) projected residual are independently checked with
80-digit analytic whole-line Gaussian correlations. No old records change.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
TOL = 1e-12


def entire(y):
    small = y < 1e-2
    ys = np.where(small, 1., y)
    r = np.sqrt(ys)
    c = np.where(small, 1+y/2+y*y/24+y**3/720+y**4/40320+y**5/3628800, np.cosh(r))
    s = np.where(small, 1+y/6+y*y/120+y**3/5040+y**4/362880+y**5/39916800, np.sinh(r)/r)
    q = np.where(small, 1/3+y/30+y*y/840+y**3/45360+y**4/3991680+y**5/518918400,
                 (np.cosh(r)-np.sinh(r)/r)/ys)
    return c, s, q


def basis(x, beta, t):
    z = x-beta
    C, S, Q = entire(z*z*t)
    A = np.exp(-(z*z+t)/2)
    M, D = A*C, A*z*S
    Mx, Dx = -z*M+t*D, M-z*D
    Mb, Db = -Mx, -Dx
    Mxb, Dxb = M+z*Mx-t*Dx, -Mx+D+z*Dx
    Mt, Dt = (z*D-M)/2, (A*z**3*Q-D)/2
    Mxt, Dxt = -z*Mt+D+t*Dt, Mt-z*Dt
    return M, D, Mx, Dx, Mb, Db, Mxb, Dxb, Mt, Dt, Mxt, Dxt


class Problem:
    def __init__(self, family, nodes=3201, radius=16.):
        self.family = family
        self.x = np.linspace(-radius, radius, nodes)
        self.w = np.full(nodes, 2*radius/(nodes-1))
        self.w[[0, -1]] *= .5
        if family == 'ordinary':
            zp, zm = self.x-1, self.x+1
            gp, gm = np.exp(-zp*zp/2), np.exp(-zm*zm/2)
            self.u, self.ux = gp+gm, -zp*gp-zm*gm
        else:
            variance = 1. if family == 'zero_gap' else .99
            g = np.exp(-self.x*self.x/(2*variance))
            self.u = -self.x*g/variance
            self.ux = (self.x*self.x/variance**2-1/variance)*g
        self.norm2 = self.ip(self.u, self.ux, self.u, self.ux)

    def ip(self, a, ax, b, bx):
        return float(np.sum(self.w*(a*b+ax*bx)))

    def evaluate(self, beta, t):
        M,D,Mx,Dx,Mb,Db,Mxb,Dxb,Mt,Dt,Mxt,Dxt = basis(self.x,beta,t)
        K = np.array([[self.ip(M,Mx,M,Mx), self.ip(M,Mx,D,Dx)],
                      [self.ip(M,Mx,D,Dx), self.ip(D,Dx,D,Dx)]])
        rhs = np.array([self.ip(M,Mx,self.u,self.ux), self.ip(D,Dx,self.u,self.ux)])
        a,c = np.linalg.solve(K,rhs)
        r,rx = a*M+c*D-self.u, a*Mx+c*Dx-self.ux
        F = self.ip(r,rx,r,rx)/(2*self.norm2)
        grad = np.array([self.ip(r,rx,a*Mb+c*Db,a*Mxb+c*Dxb),
                         self.ip(r,rx,a*Mt+c*Dt,a*Mxt+c*Dxt)])/self.norm2
        return dict(F=F, shifted=-float(rhs@np.array([a,c]))/(2*self.norm2),
                    gradient=grad, a=float(a), c=float(c))


def projected_residual(beta,t,gradient,floor=0.):
    z = np.array([beta,t])
    trial = np.clip(z-gradient,[-3.,floor],[3.,4.])
    return float(np.linalg.norm(z-trial))


def exact_check(family,beta,t,a,c):
    with mp.workdps(80):
        beta,t,a,c = [mp.mpf(float(v)) for v in (beta,t,a,c)]
        variance = mp.mpf(1) if family != 'positive_gap' else mp.mpf('.99')
        def kernel(q, s=mp.mpf(1)):
            total=1+s
            return mp.sqrt(2*mp.pi*s/total)*mp.exp(-q*q/(2*total))*(1+1/total-q*q/total**2)
        def corr(q,t,s=mp.mpf(1)):
            total=1+s
            root=mp.sqrt(t)
            z=q*root/total
            co=mp.cosh(z)
            si=q/total if t==0 else mp.sinh(z)/root
            base=mp.sqrt(2*mp.pi*s/total)*mp.exp(-(q*q+t)/(2*total))
            poly=1+1/total-(q*q+t)/total**2
            return mp.re(base*(poly*co+2*q*t*si/total**2)),mp.re(base*(-poly*si-2*q*co/total**2))
        def gram(t):
            ex=mp.exp(-t)
            MM=mp.sqrt(mp.pi)*(mp.mpf('1.5')+ex*(mp.mpf('1.5')-t))/2
            DD=mp.sqrt(mp.pi)*mp.mpf('1.25') if t==0 else mp.sqrt(mp.pi)*(-mp.mpf('1.5')*mp.expm1(-t)+t*ex)/(2*t)
            return MM,DD
        def rhs(beta,t):
            if family=='ordinary':
                left,right=corr(beta-1,t),corr(beta+1,t)
                return left[0]+right[0],left[1]+right[1]
            return tuple(mp.diff(lambda q:corr(q,t,variance)[j],beta) for j in (0,1))
        if family=='ordinary':
            norm2=2*kernel(0)+2*kernel(2)
        else:
            norm2=mp.sqrt(mp.pi)*(1/(2*mp.sqrt(variance))+3/(4*variance**mp.mpf('1.5')))
        MM,DD=gram(t)
        bM,bD=rhs(beta,t)
        F=(norm2+a*a*MM+c*c*DD-2*a*bM-2*c*bD)/(2*norm2)
        af,cf=bM/MM,bD/DD
        gb=-(af*mp.diff(lambda q:rhs(q,t)[0],beta)+cf*mp.diff(lambda q:rhs(q,t)[1],beta))/norm2
        gt=(af*af*mp.diff(lambda q:gram(q)[0],t)/2+cf*cf*mp.diff(lambda q:gram(q)[1],t)/2
            -af*mp.diff(lambda q:rhs(beta,q)[0],t)-cf*mp.diff(lambda q:rhs(beta,q)[1],t))/norm2
        return dict(error=float(mp.sqrt(max(mp.mpf(0),2*F))),
                    canonical_residual=projected_residual(float(beta),float(t),np.array([float(gb),float(gt)])),
                    gradient_beta=float(gb),gradient_t=float(gt),coefficient_residual=float(mp.sqrt((a*MM-bM)**2+(c*DD-bD)**2)/norm2))


class ResidualStop(Exception):
    def __init__(self,q): self.q=q.copy()


def run(problem,seed,chart,floor,objective):
    rng=np.random.default_rng(20261112+seed)
    beta=float(rng.uniform(-.08,.08))
    h=float(rng.uniform(.7,1.3) if problem.family=='ordinary' else rng.uniform(.05,.3))
    initial=np.array([beta,h if chart=='h' else h*h])
    calls,iterations=0,0
    trace=[]
    cached={}
    def evaluate(q):
        nonlocal calls
        key=tuple(q)
        if key not in cached:
            calls+=1
            t=float(q[1]**2 if chart=='h' else q[1])
            cached.clear()
            cached[key]=problem.evaluate(float(q[0]),t)
        return cached[key]
    def unpack(q): return float(q[0]),float(q[1]**2 if chart=='h' else q[1])
    def fun(q):
        val=evaluate(q)
        grad=val['gradient'].copy()
        if chart=='h':grad[1]*=2*q[1]
        return val['shifted'] if objective=='shifted_gram' else val['F'],grad
    def callback(q):
        nonlocal iterations
        iterations+=1
        val=evaluate(q)
        b,t=unpack(q)
        R=projected_residual(b,t,val['gradient'])
        Rf=projected_residual(b,t,val['gradient'],floor=floor)
        trace.append(dict(iteration=iterations,fits=calls,beta=b,t=t,F=val['F'],canonical_residual=R,feasible_t_residual=Rf))
        if objective=='residual' and Rf<=TOL:raise ResidualStop(q)
    started=time.perf_counter()
    lower=math.sqrt(floor) if chart=='h' else floor
    upper=2. if chart=='h' else 4.
    try:
        opt=minimize(fun,initial,jac=True,method='L-BFGS-B',callback=callback,
                     bounds=[(-3.,3.),(lower,upper)],
                     options=dict(ftol=1e-15 if objective=='shifted_gram' else 0.,
                                  gtol=1e-12 if objective=='shifted_gram' else 0.,
                                  maxiter=20000,maxfun=50000,maxls=40,maxcor=20))
        q=opt.x
        stop=str(opt.message)
    except ResidualStop as done:
        q=done.q
        stop='common feasible t residual <= 1e-12'
    seconds=time.perf_counter()-started
    final=evaluate(q)
    b,t=unpack(q)
    exact=exact_check(problem.family,b,t,final['a'],final['c'])
    refined=Problem(problem.family,nodes=6401,radius=20.).evaluate(b,t)
    return dict(family=problem.family,seed=seed,chart=chart,t_floor=floor,objective=objective,
                initial_beta=beta,initial_h=h,beta=b,t=t,coefficients=[final['a'],final['c']],
                iterations=iterations,fits=calls,seconds=seconds,stop=stop,
                error_grid=math.sqrt(2*final['F']),feasible_t_residual=projected_residual(b,t,final['gradient'],floor),
                refined_objective_difference=abs(final['F']-refined['F']),
                refined_gradient_difference=float(np.linalg.norm(final['gradient']-refined['gradient'])),
                **exact,trace=trace)


def summarize(records):
    output=[]
    keys=sorted({(r['family'],r['objective'],r['chart'],r['t_floor']) for r in records})
    for family,objective,chart,floor in keys:
        rows=[r for r in records if (r['family'],r['objective'],r['chart'],r['t_floor'])==(family,objective,chart,floor)]
        row=dict(family=family,objective=objective,chart=chart,t_floor=floor,n=len(rows))
        for key in ('error','canonical_residual','feasible_t_residual','iterations','fits','seconds','t'):
            row['median_'+key]=float(np.median([r[key] for r in rows]))
            row['max_'+key]=float(max(r[key] for r in rows))
        row['canonical_successes']=sum(r['canonical_residual']<=1.01*TOL for r in rows)
        row['boundary_hits']=sum(r['t']==floor for r in rows)
        output.append(row)
    return output


def save_csv(path,records):
    fields=list(records[0])
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--starts',type=int,default=10)
    parser.add_argument('--output',type=Path,default=HERE)
    parser.add_argument('--family',choices=['all','zero_gap','positive_gap','ordinary'],default='all')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    families=['zero_gap','positive_gap','ordinary'] if args.family=='all' else [args.family]
    design=[('h',1e-24,'shifted_gram'),('t',0.,'shifted_gram'),('h',1e-24,'residual')]
    design += [('t',floor,'residual') for floor in (0.,1e-4,1e-8,1e-12,1e-24)]
    records=[]
    started=time.perf_counter()
    for family in families:
        problem=Problem(family)
        for seed in range(args.starts):
            for chart,floor,objective in design:
                record=run(problem,seed,chart,floor,objective)
                records.append(record)
            print(f'{family} seed {seed}: completed {len(records)} runs',flush=True)
    metadata=dict(protocol='Exploratory local controls fixed in this script before the reported run.',
                  version=1,starts=args.starts,seed_base=20261112,
                  families=families,design=design,tolerance=TOL,
                  grid=dict(nodes=3201,interval=[-16,16],rule='trapezoidal'),
                  validation_grid=dict(nodes=6401,interval=[-20,20]),
                  exact_validation_digits=80,
                  stopping='Residual objective: ftol=gtol=0 and common feasible canonical t residual <=1e-12; shifted Gram: ftol=1e-15, chart gtol=1e-12.',
                  timing='One serial measurement per run with library threads fixed to one. Timings include optimizer and diagnostics used during iterations, exclude endpoint high-precision and refinement validation. Host idle state is not certified; times are descriptive only.',
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,
                  machine=platform.machine(),system=platform.platform(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  total_seconds=time.perf_counter()-started)
    (args.output/'results.json').write_text(json.dumps(dict(metadata=metadata,records=records),indent=2)+'\n')
    flat=[{k:v for k,v in r.items() if k not in ('trace','coefficients')} for r in records]
    save_csv(args.output/'runs.csv',flat)
    summary=summarize(records)
    save_csv(args.output/'summary.csv',summary)
    for row in summary:
        print(f"{row['family']:12s} {row['objective']:12s} {row['chart']} floor={row['t_floor']:g} "
              f"E={row['median_error']:.4g} R={row['median_canonical_residual']:.3g} "
              f"fits={row['median_fits']:g} seconds={row['median_seconds']:.4g}")


if __name__=='__main__':main()
