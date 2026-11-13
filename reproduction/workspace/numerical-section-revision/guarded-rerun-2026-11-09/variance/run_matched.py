#!/usr/bin/env python3
"""New matched raw/stable/completed variance study; preserves historical records."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits
import math_support as exact

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
X = np.linspace(-16., 16., 3201)
W = np.full_like(X, .01)
W[[0, -1]] *= .5
H_MIN = 1e-12
K0 = 1.5 * math.sqrt(math.pi)
KAPPA2 = math.sqrt(21 * math.sqrt(math.pi) / 8)
KAPPA3 = math.sqrt(135 * math.sqrt(math.pi) / 16)


def ip(v, dv, w, dw):
    return float(np.sum(W * (v*w + dv*dw)))


def initial(seed):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0., 1., 2), rng.uniform(-1.5, 1.5, 2)]


def raw_to_stable(p):
    ix = np.argsort(p[2:])
    wm, wp = p[:2][ix]
    bm, bp = p[2:][ix]
    h = (bp-bm)/2
    return np.array([wm+wp, h*(wp-wm), (bp+bm)/2, h])


def basis(beta, h):
    """E,D and their x,beta,h derivatives, cancellation-free for h near zero."""
    z = X-beta
    if h > .05:
        zp, zm = z-h, z+h
        gp, gm = np.exp(-zp*zp/2), np.exp(-zm*zm/2)
        e, d = (gp+gm)/2, (gp-gm)/(2*h)
        ex, dx = (-zp*gp-zm*gm)/2, (-zp*gp+zm*gm)/(2*h)
        eb, db = -ex, -dx
        exb = ((1-zp*zp)*gp+(1-zm*zm)*gm)/2
        dxb = ((1-zp*zp)*gp-(1-zm*zm)*gm)/(2*h)
        eh = (zp*gp-zm*gm)/2
        dh = ((zp*gp+zm*gm)/2-d)/h
        exh = ((1-zp*zp)*gp-(1-zm*zm)*gm)/2
        dxh = (exb-dx)/h
        return e,d,ex,dx,eb,db,exb,dxb,eh,dh,exh,dxh
    # Outside |z|<=40 these terms are below the double-precision range.
    active = abs(z) <= 40
    zz = np.where(active, z, 0.)
    y = zz*h
    y2 = y*y
    # sinhc and (y*cosh(y)-sinh(y))/y^3 have regular even expansions.
    small = abs(y) < .1
    ys = np.where(small, 1., y)
    sinhc = np.where(small, 1+y2/6+y2*y2/120+y2**3/5040+y2**4/362880,
                     np.sinh(ys)/ys)
    q = np.where(small, 1/3+y2/30+y2*y2/840+y2**3/45360+y2**4/3991680,
                 (ys*np.cosh(ys)-np.sinh(ys))/(ys**3))
    a = np.where(active, np.exp(-(zz*zz+h*h)/2), 0.)
    e, d = a*np.cosh(y), a*zz*sinhc
    ex, dx = -zz*e+h*h*d, e-zz*d
    eb, db = -ex, -dx
    exb, dxb = e+zz*ex-h*h*dx, -ex+d+zz*dx
    eh = h*(zz*d-e)
    dh = a*h*(-zz*sinhc+zz**3*q)
    exh, dxh = -zz*eh+2*h*d+h*h*dh, eh-zz*dh
    return e,d,ex,dx,eb,db,exb,dxb,eh,dh,exh,dxh


def state(p, completed=False):
    a,c,beta = p[:3]
    h = 0. if completed else p[3]
    e,d,ex,dx,eb,db,exb,dxb,eh,dh,exh,dxh = basis(beta,h)
    v,dv = a*e+c*d, a*ex+c*dx
    derivative_pairs = [(e,ex), (d,dx), (a*eb+c*db,a*exb+c*dxb)]
    if not completed:
        derivative_pairs.append((a*eh+c*dh,a*exh+c*dxh))
    return v,dv,derivative_pairs


class Complete(Exception):
    def __init__(self,p):
        self.p = p.copy()


def run_one(s, seed, method):
    gu = np.exp(-X*X/(2*s))
    u,du = -X/s*gu, (X*X/s**2-1/s)*gu
    n2 = ip(u,du,u,du)
    start=time.perf_counter()
    p0 = initial(seed)
    p = p0.copy() if method == 'S' else raw_to_stable(p0)
    calls=iterations=persistent=unchanged=0
    completed=False
    trace,scans,events,returns = [],[],[],[]

    def evaluate(q):
        if method == 'S':
            w,b = q[:2],q[2:]
            z = X[None,:]-b[:,None]
            g = np.exp(-z*z/2)
            dg,ddg = -z*g,(z*z-1)*g
            v,dv=w@g,w@dg
            derivatives=[(g[i],dg[i]) for i in range(2)]
            derivatives += [(-w[i]*dg[i],-w[i]*ddg[i]) for i in range(2)]
        else:
            v,dv,derivatives = state(q,completed)
        r,dr=v-u,dv-du
        energy=ip(r,dr,r,dr)/2
        grad=np.array([ip(r,dr,a,b)/n2 for a,b in derivatives])
        return energy/n2,grad,v,dv

    def loss(q):
        nonlocal calls
        calls += 1
        answer=evaluate(q)
        return answer[:2]

    def parameters(q):
        qstable=raw_to_stable(q) if method=='S' else np.r_[q,0.] if completed else q
        a,c,beta,h=map(float,qstable)
        raw_norm=math.sqrt((a*a+(c/h)**2)/2) if h>0 else None
        return dict(a=a,c=c,beta=beta,h=h,stable_coefficient_norm=math.hypot(a,c),
                    ordinary_weight_norm=raw_norm,
                    stored_coefficient_norm=raw_norm if method=='S' else math.hypot(a,c))

    def record(q):
        energy,grad,_,_=evaluate(q)
        trace.append(dict(iteration=iterations,objective_evaluations=calls,
                          elapsed_seconds=time.perf_counter()-start,
                          phase='completed' if completed else 'ordinary' if method=='S' else 'stable',
                          relative_error=math.sqrt(max(0.,2*energy)),
                          normalized_energy_gap=energy,gradient_norm=float(np.linalg.norm(grad)),
                          **parameters(q)))

    def scan(q,trigger):
        nonlocal persistent
        if completed or method=='S':
            return False
        a,c,beta,h=map(float,q)
        cross=math.sqrt(math.pi)*math.exp(-h*h)*(1.5-h*h)
        gram=math.sqrt(max(0.,1-abs(cross)/K0))
        flag=bool(2*h<=.5 and gram<=.1)
        persistent=persistent+1 if flag else 0
        scans.append(dict(iteration=iterations,trigger=trigger,half_gap=h,
                          gram_diagnostic=gram,flag=flag,persistent_flags=persistent))
        if persistent<2:
            return False
        energy,_,v,dv=evaluate(q)
        vt,dvt,_=state(q[:3],True)
        omega=h*h*(abs(a)*KAPPA2/2+abs(c)*KAPPA3/6)
        residual=math.sqrt(max(0.,2*energy*n2))
        norm=math.sqrt(ip(v,dv,v,dv))
        bound=residual*omega+.5*omega*omega
        eligible=bool(omega<=.01*norm and bound<=.0005*n2)
        after=ip(vt-u,dvt-du,vt-u,dvt-du)/2
        events.append(dict(iteration=iterations,objective_evaluations=calls,
                           elapsed_seconds=time.perf_counter()-start,
                           eligible=eligible,accepted=eligible and method=='C',
                           before=parameters(q),state_error_bound=omega,
                           state_error_measured=math.sqrt(ip(vt-v,dvt-dv,vt-v,dvt-dv)),
                           state_tolerance=.01*norm,energy_increase_bound=bound,
                           energy_allowance=.0005*n2,energy_before=energy*n2,
                           energy_after=after,energy_change=after-energy*n2))
        return eligible and method=='C'

    def callback(q):
        nonlocal iterations
        iterations+=1
        record(q)
        if iterations%20==0 and scan(q,'periodic'):
            raise Complete(q)

    record(p)
    while unchanged<2 and iterations<20000 and calls<50000:
        try:
            bounds=None if method=='S' or completed else [(None,None)]*3+[(H_MIN,None)]
            opt=minimize(loss,p,jac=True,method='L-BFGS-B',callback=callback,bounds=bounds,
                         options=dict(maxiter=20000-iterations,maxfun=50000-calls,
                                      ftol=0.,gtol=0.,maxcor=20,maxls=30))
            p=opt.x
            returns.append(dict(success=bool(opt.success),status=int(opt.status),
                                message=str(opt.message),iterations=int(opt.nit),
                                function_evaluations=int(opt.nfev),completed=completed))
            record(p)
            if scan(p,'local_return'):
                raise Complete(p)
            unchanged+=1
        except Complete as event:
            p=event.p[:3].copy()
            completed=True
            unchanged=0
            record(p)
    elapsed=time.perf_counter()-start
    final=parameters(p)
    final['optimized_parameters']=p.tolist()
    final['representation']='completed' if completed else 'ordinary' if method=='S' else 'stable finite-gap'
    # High precision validation is deliberately outside the timed optimizer scope.
    validation_started=time.perf_counter()
    with mp.workdps(80):
        check=exact.pair_evaluation(mp.mpf(str(s)),*[mp.mpf(final[k]) for k in ('beta','h','a','c')])
        final['whole_space_80digit']={k:mp.nstr(v,65) if v is not None else None for k,v in check.items()}
        final['relative_error']=float(check['relative_error'])
        final['energy_gap']=float(check['energy_gap'])
    return dict(variance=s,seed=seed,method=method,initial_parameters=p0.tolist(),
                target_norm_squared=n2,iterations=iterations,objective_evaluations=calls,
                optimizer_wall_seconds=elapsed,validation_wall_seconds=time.perf_counter()-validation_started,
                final=final,trace=trace,scans=scans,events=events,optimizer_returns=returns,
                completion_count=sum(e['accepted'] for e in events),
                stop='work_limit' if iterations>=20000 or calls>=50000 else returns[-1]['message'],
                hit_half_gap_floor=not completed and method!='S' and final['h']<=H_MIN*(1+1e-12))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variances',nargs='+',type=float,default=[.99,.995,.999])
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--methods',nargs='+',choices=['S','NC','C'],default=['S','NC','C'])
    parser.add_argument('--output',default='results')
    parser.add_argument('--timed',action='store_true',help='Acquire the shared exclusive benchmark lock.')
    args=parser.parse_args()
    output=HERE/args.output
    output.mkdir(parents=True,exist_ok=True)
    lock=(HERE.parent/'timing.lock').open('a+') if args.timed else None
    if lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        with threadpool_limits(limits=1):
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__),HERE/'math_support.py')}
            meta=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,
                      platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
                      threadpool_info=threadpool_info(),source_sha256=sources,
                      quadrature=dict(rule='trapezoid',nodes=3201,interval=[-16,16]),
                      half_gap_floor=H_MIN,
                      stopping='two local returns without structural change; total 20000 iterations/50000 calls; ftol=gtol=0',
                      completion='conditioning-only detection, every20 iterations and local return; two consecutive flags; same state/energy bound as historical experiment',
                      coefficient_policy='joint coefficient/geometry descent for all methods, including after completion; no variable projection',
                      runtime_scope='initialization/representation setup and full optimizer including tracing, scans, transfer tests and restarts; excludes target/quadrature generation, 80-digit endpoint validation and output serialization',
                      historical_source='positive-gap-persistence/variance-family/results.json',
                      historical_source_sha256=hashlib.sha256((ROOT/'positive-gap-persistence/variance-family/results.json').read_bytes()).hexdigest(),
                      command=sys.argv,exclusive_timing_lock=bool(lock),
                      run_order='rotate S/NC/C order by (seed+variance_index) modulo3')
            (output/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
            for j,s in enumerate(args.variances):
                for seed in args.seeds:
                    methods=args.methods[(seed+j)%len(args.methods):]+args.methods[:(seed+j)%len(args.methods)]
                    for method in methods:
                        path=output/f's{s:g}_seed{seed}_{method}.json'
                        if path.exists():
                            raise FileExistsError(f'Refusing to overwrite {path}')
                        row=run_one(s,seed,method)
                        path.write_text(json.dumps(row,indent=1)+'\n')
                        print(f's={s:g} seed={seed} {method}: E={row["final"]["relative_error"]:.9g} '
                              f'wall={row["optimizer_wall_seconds"]:.3f}s iterations={row["iterations"]} '
                              f'completions={row["completion_count"]} floor={row["hit_half_gap_floor"]}',flush=True)
    finally:
        if lock:
            fcntl.flock(lock,fcntl.LOCK_UN)
            lock.close()


if __name__=='__main__':
    main()
