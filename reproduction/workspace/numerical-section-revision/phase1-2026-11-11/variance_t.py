#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): the Gaussian-variance family with the revised Algorithm 1 (pair in t = h^2 >= 0).

Imports the frozen matched runner (guarded-rerun-2026-11-09/variance/run_matched.py) unchanged and adds the
method "T": the pair is optimized in (a, c, beta, t) with t >= 0, using the frozen cancellation-free basis.
The t-derivatives are the frozen h-derivatives divided by 2h; on the frozen small-h branch (h <= 0.05) every
h-derivative carries an explicit factor h, so the quotients are evaluated without division and stay exact at
t = 0.  Everything else (quadrature, starts, stopping rule, tolerances, joint coefficient/geometry descent,
80-digit endpoint check) is the frozen protocol; there is no discrete completion.
Usage: python3.14 variance_t.py --output runs/variance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import mpmath as mp
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'guarded-rerun-2026-11-09' / 'variance'
sys.path.insert(0, str(FROZEN))
import run_matched as rm          # noqa: E402  (frozen)
import math_support as exact      # noqa: E402  (frozen)


def basis_t(beta, t):
    """E, D, their x/beta derivatives and their t-derivatives, smooth through t = 0."""
    h = math.sqrt(max(t, 0.0))
    e, d, ex, dx, eb, db, exb, dxb, eh, dh, exh, dxh = rm.basis(beta, h)
    if h > .05:                                   # frozen large-h branch: divide numerically
        et, dt, ext, dxt = eh / (2 * h), dh / (2 * h), exh / (2 * h), dxh / (2 * h)
        return e, d, ex, dx, eb, db, exb, dxb, et, dt, ext, dxt
    z = rm.X - beta                               # frozen small-h branch, explicit factor h removed
    active = abs(z) <= 40
    zz = np.where(active, z, 0.)
    y = zz * h
    y2 = y * y
    small = abs(y) < .1
    ys = np.where(small, 1., y)
    sinhc = np.where(small, 1 + y2 / 6 + y2 * y2 / 120 + y2 ** 3 / 5040 + y2 ** 4 / 362880, np.sinh(ys) / ys)
    q = np.where(small, 1 / 3 + y2 / 30 + y2 * y2 / 840 + y2 ** 3 / 45360 + y2 ** 4 / 3991680,
                 (ys * np.cosh(ys) - np.sinh(ys)) / (ys ** 3))
    a = np.where(active, np.exp(-(zz * zz + h * h) / 2), 0.)
    et = (zz * d - e) / 2                         # eh = h (zz d - e)
    dt = a * (-zz * sinhc + zz ** 3 * q) / 2      # dh = a h (-zz sinhc + zz^3 q)
    ext = -zz * et + d + h * h * dt               # exh = -zz eh + 2 h d + h^2 dh
    dxt = et - zz * dt                            # dxh = eh - zz dh
    return e, d, ex, dx, eb, db, exb, dxb, et, dt, ext, dxt


def run_one_t(s, seed):
    X = rm.X
    gu = np.exp(-X * X / (2 * s))
    u, du = -X / s * gu, (X * X / s ** 2 - 1 / s) * gu
    n2 = rm.ip(u, du, u, du)
    start = time.perf_counter()
    p0 = rm.initial(seed)
    st = rm.raw_to_stable(p0)
    p = np.r_[st[:3], st[3] ** 2]                 # (a, c, beta, t = h^2)
    calls = iterations = unchanged = 0
    trace, returns = [], []

    def evaluate(q):
        a, c, beta, t = q
        e, d, ex, dx, eb, db, exb, dxb, et, dt, ext, dxt = basis_t(beta, t)
        v, dv = a * e + c * d, a * ex + c * dx
        derivatives = [(e, ex), (d, dx), (a * eb + c * db, a * exb + c * dxb), (a * et + c * dt, a * ext + c * dxt)]
        r, dr = v - u, dv - du
        energy = rm.ip(r, dr, r, dr) / 2
        grad = np.array([rm.ip(r, dr, x, y) / n2 for x, y in derivatives])
        return energy / n2, grad

    def loss(q):
        nonlocal calls
        calls += 1
        return evaluate(q)

    def parameters(q):
        a, c, beta, t = map(float, q)
        h = math.sqrt(max(t, 0.))
        raw = math.sqrt((a * a + (c / h) ** 2) / 2) if h > 0 else None
        return dict(a=a, c=c, beta=beta, t=t, h=h, stable_coefficient_norm=math.hypot(a, c),
                    ordinary_weight_norm=raw, stored_coefficient_norm=math.hypot(a, c))

    def record(q):
        energy, grad = evaluate(q)
        trace.append(dict(iteration=iterations, objective_evaluations=calls,
                          elapsed_seconds=time.perf_counter() - start, phase='t-chart',
                          relative_error=math.sqrt(max(0., 2 * energy)), normalized_energy_gap=energy,
                          gradient_norm=float(np.linalg.norm(grad)), **parameters(q)))

    def callback(q):
        nonlocal iterations
        iterations += 1
        record(q)

    record(p)
    while unchanged < 2 and iterations < 20000 and calls < 50000:   # frozen stopping rule
        opt = minimize(loss, p, jac=True, method='L-BFGS-B', callback=callback,
                       bounds=[(None, None)] * 3 + [(0., None)],
                       options=dict(maxiter=20000 - iterations, maxfun=50000 - calls,
                                    ftol=0., gtol=0., maxcor=20, maxls=30))
        p = opt.x
        returns.append(dict(success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                            iterations=int(opt.nit), function_evaluations=int(opt.nfev)))
        record(p)
        unchanged += 1                            # no structural change exists in the t-chart
    elapsed = time.perf_counter() - start
    final = parameters(p)
    final['optimized_parameters'] = p.tolist()
    final['representation'] = 'pair at t = 0 (derivative limit)' if final['t'] == 0. else 'pair, t > 0'
    validation_started = time.perf_counter()
    with mp.workdps(80):
        check = exact.pair_evaluation(mp.mpf(str(s)), *[mp.mpf(final[k]) for k in ('beta', 'h', 'a', 'c')])
        final['whole_space_80digit'] = {k: mp.nstr(v, 65) if v is not None else None for k, v in check.items()}
        final['relative_error'] = float(check['relative_error'])
        final['energy_gap'] = float(check['energy_gap'])
    # projected gradient at the end (t bound only)
    _, g = evaluate(p)
    pg = g.copy()
    if p[3] <= 0. and g[3] > 0:
        pg[3] = 0.
    return dict(variance=s, seed=seed, method='C', method_detail='pair in t = h^2 >= 0 (revised Algorithm 1)',
                initial_parameters=p0.tolist(), target_norm_squared=n2, iterations=iterations,
                objective_evaluations=calls, optimizer_wall_seconds=elapsed,
                validation_wall_seconds=time.perf_counter() - validation_started, final=final, trace=trace,
                optimizer_returns=returns, zero_gap=bool(final['t'] == 0.),
                projected_gradient_max_abs=float(np.max(np.abs(pg))),
                stop='work_limit' if iterations >= 20000 or calls >= 50000 else returns[-1]['message'],
                timing_context='shared host during Phase 1 campaigns; not a controlled timing')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variances', nargs='+', type=float, default=[.99, .995, .999])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--output', type=Path, default=HERE / 'runs' / 'variance')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (Path(__file__), FROZEN / 'run_matched.py', FROZEN / 'math_support.py')}
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        python=sys.version, numpy=np.__version__, source_sha256=sources, command=sys.argv,
        method='C: pair in t = h^2 >= 0, frozen basis, frozen stopping rule (two local returns), ftol = gtol = 0'),
        indent=2) + '\n')
    with threadpool_limits(limits=1):
        for s in args.variances:
            for seed in args.seeds:
                path = args.output / f's{s:g}_seed{seed}_C.json'
                if path.exists():
                    raise FileExistsError(path)
                row = run_one_t(s, seed)
                path.write_text(json.dumps(row, indent=1) + '\n')
                print(f"s={s:g} seed={seed} C: E={row['final']['relative_error']:.9g} t={row['final']['t']:.3g} "
                      f"iter={row['iterations']} calls={row['objective_evaluations']} pg={row['projected_gradient_max_abs']:.2e}",
                      flush=True)


if __name__ == '__main__':
    main()
