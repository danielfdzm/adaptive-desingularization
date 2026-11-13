#!/usr/bin/env python3
"""Variable-projection version of the step-10 toy (added after the joint-protocol sweep; not pre-registered).

The prediction of step 6 concerns the reduced energy: the coefficients are refitted at every geometry,
as in Algorithm 1. Here L-BFGS-B optimizes only the geometry -- (beta, h) for NC and C before
completion, (beta, t) with t = h^2 for T, beta after completion -- and the two coefficients solve the
2x2 normal equations exactly at every evaluation. The objective is the normalized squared H1 error
0.5 ||u - v||^2 / ||u||^2, evaluated from the residual on the matched 3201-node grid (no cancellation),
and its gradient follows from the envelope theorem. Initial geometry, detection and acceptance rules,
stopping rule (two local returns without structural change) and bounds are those of toy.py.
"""
from __future__ import annotations

import math
import time

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

import toy

X, ip = toy.X, toy.ip


def run_one(s, seed, method, ftol=0., gtol=0., max_iterations=20000, max_calls=50000):
    gu = np.exp(-X*X/(2*s))
    u, du = -X/s*gu, (X*X/s**2-1/s)*gu
    n2 = ip(u, du, u, du)
    start = time.perf_counter()
    q0 = toy.raw_to_stable(toy.initial(seed))
    beta0, h0 = float(q0[2]), float(q0[3])
    p = np.array([beta0, h0*h0 if method == 'T' else h0])
    chart = 't' if method == 'T' else 'h'
    completed = False
    calls = iterations = persistent = unchanged = 0
    trace, events, returns = [], [], []

    def fitted(q):
        g = 0. if completed else q[1]
        e, d, ex, dx, eb, db, exb, dxb, eg, dg, exg, dxg = toy.basis_t(q[0], g) if chart == 't' else toy.basis(q[0], g)
        M = np.array([[ip(e, ex, e, ex), ip(e, ex, d, dx)], [ip(d, dx, e, ex), ip(d, dx, d, dx)]])
        b = np.array([ip(e, ex, u, du), ip(d, dx, u, du)])
        a, c = np.linalg.solve(M, b)
        v, dv = a*e+c*d, a*ex+c*dx
        r, dr = v-u, dv-du
        energy = ip(r, dr, r, dr)/2/n2
        grad = [ip(r, dr, a*eb+c*db, a*exb+c*dxb)/n2]
        if not completed:
            grad.append(ip(r, dr, a*eg+c*dg, a*exg+c*dxg)/n2)
        return energy, np.array(grad), a, c, v, dv

    def loss(q):
        nonlocal calls
        calls += 1
        energy, grad, *_ = fitted(q)
        return energy, grad

    def parameters(q):
        energy, grad, a, c, _, _ = fitted(q)
        if completed:
            h = t = 0.
        elif chart == 't':
            t = float(q[1]); h = math.sqrt(t)
        else:
            h = float(q[1]); t = h*h
        return dict(beta=float(q[0]), h=h, t=t, a=float(a), c=float(c), relative_error_double=math.sqrt(max(0., 2*energy)),
                    gradient_max_abs=float(np.max(np.abs(grad))), gradient=grad.tolist(),
                    ordinary_weight_norm=math.sqrt((a*a+(c/h)**2)/2) if h > 0 else None)

    def record(q):
        trace.append(dict(iteration=iterations, objective_evaluations=calls,
                          phase='completed' if completed else 'stable', **parameters(q)))

    def scan(q):
        nonlocal persistent
        if completed or method != 'C':
            return False
        h = float(q[1])
        cross = math.sqrt(math.pi)*math.exp(-h*h)*(1.5-h*h)
        gram = math.sqrt(max(0., 1-abs(cross)/toy.K0))
        flag = bool(2*h <= .5 and gram <= .1)
        persistent = persistent+1 if flag else 0
        if persistent < 2:
            return False
        energy, _, a, c, v, dv = fitted(q)
        e, d, ex, dx, *_ = toy.basis(q[0], 0.)
        vt, dvt = a*e+c*d, a*ex+c*dx
        omega = h*h*(abs(a)*toy.KAPPA2/2+abs(c)*toy.KAPPA3/6)
        residual = math.sqrt(max(0., 2*energy*n2))
        norm = math.sqrt(ip(v, dv, v, dv))
        bound = residual*omega+.5*omega*omega
        ok = bool(omega <= .01*norm and bound <= .0005*n2)
        events.append(dict(iteration=iterations, accepted=ok, h=h, omega=omega, bound=bound))
        return ok

    class Complete(Exception):
        pass

    def callback(q):
        nonlocal iterations
        iterations += 1
        record(q)
        if iterations % 20 == 0 and scan(q):
            raise Complete(q.copy())

    record(p)
    while unchanged < 2 and iterations < max_iterations and calls < max_calls:
        try:
            bounds = [(None, None)] if completed else [(None, None), (0., None) if chart == 't' else (toy.H_MIN, None)]
            opt = minimize(loss, p, jac=True, method='L-BFGS-B', callback=callback, bounds=bounds,
                           options=dict(maxiter=max_iterations-iterations, maxfun=max_calls-calls,
                                        ftol=ftol, gtol=gtol, maxcor=20, maxls=30))
            p = opt.x
            returns.append(dict(status=int(opt.status), message=str(opt.message), iterations=int(opt.nit),
                                function_evaluations=int(opt.nfev), completed=completed))
            record(p)
            if scan(p):
                raise Complete(p.copy())
            unchanged += 1
        except Complete as event:
            p = np.array([event.args[0][0]])
            completed = True
            unchanged = 0
            record(p)
    final = parameters(p)
    with mp.workdps(80):
        check = toy.exact.pair_evaluation(mp.mpf(str(s)), mp.mpf(final['beta']), mp.mpf(final['h']),
                                          mp.mpf(final['a']), mp.mpf(final['c']))
    final['relative_error'] = float(check['relative_error'])
    return dict(variance=s, seed=seed, method=method, protocol='variable projection', ftol=ftol, gtol=gtol,
                iterations=iterations, objective_evaluations=calls, seconds=time.perf_counter()-start,
                completions=sum(e['accepted'] for e in events), final=final, trace=trace, events=events,
                returns=returns, last_return_message=returns[-1]['message'] if returns else None,
                stop='work_limit' if iterations >= max_iterations or calls >= max_calls else returns[-1]['message'])
