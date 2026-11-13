#!/usr/bin/env python3
"""Matched Gaussian-variance protocol with the squared half-gap chart T added.

S, NC and C are those of `guarded-rerun-2026-11-09/variance/run_matched.py`
(code copied verbatim, see `run_one`); `validate_against_records` checks that
they reproduce the stored records exactly. T optimizes (a, c, beta, t) with
t = h^2 >= 0, using

    E = A cosh(z sqrt t),  D = A z S(z^2 t),  A = exp(-(z^2 + t)/2),

with z = x - beta, C(y) = cosh(sqrt y), S(y) = sinh(sqrt y)/sqrt y and
Q(y) = (C - S)/y entire. Their x, beta and t derivatives are

    E_x = -z E + t D,             D_x = E - z D,
    E_t = (z D - E)/2,            D_t = (A z^3 Q - D)/2,
    E_xt = -z E_t + D + t D_t,    D_xt = E_t - z D_t,

so E, D, their derivatives and the gradient are smooth at t = 0, where
E = G(. - beta) and D = -G'(. - beta). T never completes and needs no
acceptance test. `ftol` and `gtol` can be overridden for the tolerance sweep.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_n] = '1'
import mpmath as mp
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VAR = ROOT / 'numerical-section-revision' / 'guarded-rerun-2026-11-09' / 'variance'
sys.path.insert(0, str(VAR))
import math_support as exact          # noqa: E402  (unchanged analytic formulas)

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
    """Verbatim from run_matched.py: E,D and x,beta,h derivatives."""
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
    active = abs(z) <= 40
    zz = np.where(active, z, 0.)
    y = zz*h
    y2 = y*y
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


def entire(y):
    """C(y)=cosh(sqrt y), S(y)=sinh(sqrt y)/sqrt y, Q(y)=(C-S)/y for y >= 0."""
    small = y < 1e-2
    ys = np.where(small, 1., y)
    r = np.sqrt(ys)
    c = np.where(small, 1+y/2+y*y/24+y**3/720+y**4/40320+y**5/3628800, np.cosh(r))
    s = np.where(small, 1+y/6+y*y/120+y**3/5040+y**4/362880+y**5/39916800, np.sinh(r)/r)
    q = np.where(small, 1/3+y/30+y*y/840+y**3/45360+y**4/3991680+y**5/518918400, (np.cosh(r)-np.sinh(r)/r)/ys)
    return c, s, q


def basis_t(beta, t):
    """E,D and x,beta,t derivatives in the squared half-gap t >= 0."""
    z = X-beta
    active = abs(z) <= 40
    zz = np.where(active, z, 0.)
    c, s, q = entire(zz*zz*t)
    a = np.where(active, np.exp(-(zz*zz+t)/2), 0.)
    e, d = a*c, a*zz*s
    ex, dx = -zz*e+t*d, e-zz*d
    eb, db = -ex, -dx
    exb, dxb = e+zz*ex-t*dx, -ex+d+zz*dx
    et = (zz*d-e)/2
    dt = (a*zz**3*q-d)/2
    ext, dxt = -zz*et+d+t*dt, et-zz*dt
    return e,d,ex,dx,eb,db,exb,dxb,et,dt,ext,dxt


def state(p, completed=False, chart='h'):
    a,c,beta = p[:3]
    g = 0. if completed else p[3]
    e,d,ex,dx,eb,db,exb,dxb,eg,dg,exg,dxg = basis_t(beta,g) if chart == 't' else basis(beta,g)
    v,dv = a*e+c*d, a*ex+c*dx
    derivative_pairs = [(e,ex), (d,dx), (a*eb+c*db,a*exb+c*dxb)]
    if not completed:
        derivative_pairs.append((a*eg+c*dg,a*exg+c*dxg))
    return v,dv,derivative_pairs


class Complete(Exception):
    def __init__(self,p):
        self.p = p.copy()


def run_one(s, seed, method, ftol=0., gtol=0., max_iterations=20000, max_calls=50000, validate=True):
    """run_matched.run_one with method T added and optional ftol/gtol (defaults: the matched protocol)."""
    gu = np.exp(-X*X/(2*s))
    u,du = -X/s*gu, (X*X/s**2-1/s)*gu
    n2 = ip(u,du,u,du)
    start=time.perf_counter()
    p0 = initial(seed)
    if method == 'S':
        p = p0.copy()
    elif method == 'T':
        q0 = raw_to_stable(p0)
        p = np.r_[q0[:3], q0[3]**2]
    else:
        p = raw_to_stable(p0)
    chart = 't' if method == 'T' else 'h'
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
            v,dv,derivatives = state(q,completed,chart)
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
        if method == 'T':
            a,c,beta = map(float,q[:3])
            t = 0. if completed else float(q[3])
            h = math.sqrt(t)
            raw_norm = math.sqrt((a*a+(c/h)**2)/2) if h > 0 else None
            return dict(a=a,c=c,beta=beta,h=h,t=t,stable_coefficient_norm=math.hypot(a,c),
                        ordinary_weight_norm=raw_norm,stored_coefficient_norm=math.hypot(a,c))
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
                          gradient_max_abs=float(np.max(np.abs(grad))),
                          **parameters(q)))

    def scan(q,trigger):
        nonlocal persistent
        if completed or method in ('S','T'):
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
    while unchanged<2 and iterations<max_iterations and calls<max_calls:
        try:
            if method=='S' or completed:
                bounds=None
            elif method=='T':
                bounds=[(None,None)]*3+[(0.,None)]
            else:
                bounds=[(None,None)]*3+[(H_MIN,None)]
            opt=minimize(loss,p,jac=True,method='L-BFGS-B',callback=callback,bounds=bounds,
                         options=dict(maxiter=max_iterations-iterations,maxfun=max_calls-calls,
                                      ftol=ftol,gtol=gtol,maxcor=20,maxls=30))
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
    final['representation']=('completed' if completed else 'ordinary' if method=='S'
                             else 'squared half-gap chart' if method=='T' else 'stable finite-gap')
    if validate:
        with mp.workdps(80):
            check=exact.pair_evaluation(mp.mpf(str(s)),*[mp.mpf(final[k]) for k in ('beta','h','a','c')])
            final['relative_error']=float(check['relative_error'])
            final['energy_gap']=float(check['energy_gap'])
    return dict(variance=s,seed=seed,method=method,ftol=ftol,gtol=gtol,initial_parameters=p0.tolist(),
                target_norm_squared=n2,iterations=iterations,objective_evaluations=calls,
                optimizer_wall_seconds=elapsed,final=final,trace=trace,scans=scans,events=events,
                optimizer_returns=returns,completion_count=sum(e['accepted'] for e in events),
                stop='work_limit' if iterations>=max_iterations or calls>=max_calls else returns[-1]['message'],
                hit_half_gap_floor=method in ('NC','C') and not completed and final['h']<=H_MIN*(1+1e-12),
                t_zero_at_end=method=='T' and final['t']==0.)


def validate_against_records(variance=.99, seeds=range(10)):
    """S, NC and C must reproduce the stored matched records exactly (iterations, parameters, errors)."""
    import json
    rows = []
    for seed in seeds:
        for method in ('S', 'NC', 'C'):
            stored = json.loads((VAR / 'results' / f's{variance:g}_seed{seed}_{method}.json').read_text())
            new = run_one(variance, seed, method)
            same = (new['iterations'] == stored['iterations']
                    and new['objective_evaluations'] == stored['objective_evaluations']
                    and new['final']['optimized_parameters'] == stored['final']['optimized_parameters']
                    and new['final']['relative_error'] == stored['final']['relative_error']
                    and new['completion_count'] == stored['completion_count'])
            rows.append(dict(variance=variance, seed=seed, method=method, identical=same))
    return rows


if __name__ == '__main__':
    import json
    out = validate_against_records()
    print(json.dumps(out))
    print('all identical:', all(r['identical'] for r in out))
