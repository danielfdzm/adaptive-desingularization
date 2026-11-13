#!/usr/bin/env python3
"""Matched raw Gaussian optimization with and without derivative completion.

The model contains two variance-one Gaussians. The target is the derivative
of a Gaussian with variance s<1. Both methods use the same joint optimization
of weights and centers until a completion is accepted. Completion then refits
the two linear coefficients and continues center optimization in aG-cG'.
This small driver isolates that representation change; it does not establish
global optimizer convergence or certify a global minimum at a given s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import minimize

import math_support as exact

HERE = Path(__file__).resolve().parent
X = np.linspace(-16.0, 16.0, 3201)
W = np.full_like(X, X[1]-X[0])
W[[0, -1]] *= 0.5
K0 = 1.5*math.sqrt(math.pi)
KAPPA2 = math.sqrt(21*math.sqrt(math.pi)/8)
KAPPA3 = math.sqrt(135*math.sqrt(math.pi)/16)


def ip(u, du, v, dv):
    return float(np.sum(W*(u*v+du*dv)))


def target(s):
    g = np.exp(-X*X/(2*s))
    return -X/s*g, (X*X/(s*s)-1/s)*g


def initial_parameters(seed):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0, 1, 2), rng.uniform(-1.5, 1.5, 2)]


def raw_state(p):
    w, b = p[:2], p[2:]
    z = X[None, :]-b[:, None]
    g = np.exp(-z*z/2)
    dg, ddg = -z*g, (z*z-1)*g
    return w@g, w@dg, g, dg, ddg


def stable_parameters(p):
    order = np.argsort(p[2:])
    wm, wp = p[:2][order]
    bm, bp = p[2:][order]
    h = float((bp-bm)/2)
    return {'beta': float((bp+bm)/2), 'h': h, 'a': float(wp+wm),
            'c': float(h*(wp-wm)), 'ordinary_weight_norm': float(np.linalg.norm(p[:2])),
            'ordinary_weight_max': float(max(abs(wm), abs(wp)))}


def completed_basis(beta):
    z = X-beta
    g = np.exp(-z*z/2)
    dg, ddg, d3g = -z*g, (z*z-1)*g, (-z**3+3*z)*g
    return g, -dg, dg, -ddg, -dg, ddg, -ddg, d3g


def completed_state(beta, a, c):
    e, d, ex, dx, *_ = completed_basis(beta)
    return a*e+c*d, a*ex+c*dx


class AcceptedCompletion(Exception):
    def __init__(self, p):
        self.p = np.array(p, copy=True)


def run_one(s, seed, method):
    u, du = target(s)
    n2 = ip(u, du, u, du)
    p0 = initial_parameters(seed)
    start = time.perf_counter()
    trajectory, scans, events = [], [], []
    calls = 0
    iterations = 0
    persistent = 0

    def loss(p):
        nonlocal calls
        calls += 1
        v, dv, g, dg, ddg = raw_state(p)
        residual, dresidual = v-u, dv-du
        energy = ip(residual, dresidual, residual, dresidual)/2
        gw = np.sum(W*(residual*g+dresidual*dg), axis=1)/n2
        gb = -p[:2]*np.sum(W*(residual*dg+dresidual*ddg), axis=1)/n2
        return energy/n2, np.r_[gw, gb]

    def record_raw(p, phase='ordinary'):
        v, dv, *_ = raw_state(p)
        energy = ip(v-u, dv-du, v-u, dv-du)/2
        par = stable_parameters(p)
        trajectory.append({'iteration': iterations, 'evaluations': calls,
                           'phase': phase, 'energy_gap': energy,
                           'relative_error': math.sqrt(2*energy/n2),
                           'stored_coefficient_norm': par['ordinary_weight_norm'],
                           'stable_coefficient_norm': math.hypot(par['a'], par['c']),
                           **par})

    def scan(p, trigger):
        nonlocal persistent
        par = stable_parameters(p)
        h, a, c, beta = par['h'], par['a'], par['c'], par['beta']
        v, dv, *_ = raw_state(p)
        norm = math.sqrt(ip(v, dv, v, dv))
        cross = math.sqrt(math.pi)*math.exp(-h*h)*(1.5-h*h)
        gram = math.sqrt(max(0.0, 1-abs(cross)/K0))
        cancellation = math.sqrt(K0)*par['ordinary_weight_norm']/max(norm, 1e-300)
        flag = bool(2*h <= 0.5 and (gram <= 0.1 or cancellation >= 10))
        persistent = persistent+1 if flag else 0
        scans.append({'iteration': iterations, 'trigger': trigger, 'half_gap': h,
                      'gram_diagnostic': gram, 'cancellation_ratio': cancellation,
                      'flag': flag, 'persistent_flags': persistent})
        if persistent < 2:
            return False
        trial, dtrial = completed_state(beta, a, c)
        error_bound = h*h*(abs(a)*KAPPA2/2+abs(c)*KAPPA3/6)
        residual_norm = math.sqrt(ip(v-u, dv-du, v-u, dv-du))
        energy_bound = residual_norm*error_bound+error_bound*error_bound/2
        allowance = 0.0005*n2
        before = residual_norm*residual_norm/2
        after = ip(trial-u, dtrial-du, trial-u, dtrial-du)/2
        state_change = math.sqrt(ip(trial-v, dtrial-dv, trial-v, dtrial-dv))
        accepted = bool(error_bound <= 0.01*norm and energy_bound <= allowance)
        events.append({'iteration': iterations, 'evaluations': calls, 'accepted': accepted,
                       'raw_parameters': np.asarray(p).tolist(), 'before': par,
                       'trial': {'beta': beta, 'h': 0.0, 'a': a, 'c': c},
                       'state_error_bound': error_bound, 'state_error_measured': state_change,
                       'state_tolerance': 0.01*norm, 'residual_norm': residual_norm,
                       'energy_increase_bound': energy_bound, 'energy_allowance': allowance,
                       'energy_before': before, 'energy_after': after,
                       'energy_change': after-before})
        return accepted

    def callback(p):
        nonlocal iterations
        iterations += 1
        record_raw(p)
        if method == 'completion' and iterations % 20 == 0 and scan(p, 'periodic'):
            raise AcceptedCompletion(p)

    record_raw(p0)
    raw_returns = []
    p = p0.copy()
    completion = None
    try:
        # A second local call lets a pending persistent flag be checked again.
        for cycle in range(2 if method == 'completion' else 1):
            res = minimize(loss, p, jac=True, method='L-BFGS-B', callback=callback,
                           options={'maxiter': 20000, 'maxfun': 50000, 'ftol': 0.0,
                                    'gtol': 0.0, 'maxcor': 20, 'maxls': 30})
            p = res.x
            raw_returns.append({'success': bool(res.success), 'status': int(res.status),
                                'message': str(res.message), 'iterations': int(res.nit),
                                'function_evaluations': int(res.nfev)})
            if method == 'completion' and scan(p, 'local_return'):
                raise AcceptedCompletion(p)
            if method == 'ordinary' or persistent == 0:
                break
        final = {'representation': 'ordinary', 'raw_parameters': p.tolist(), **stable_parameters(p)}
        stop = raw_returns[-1]['message']
    except AcceptedCompletion as accepted:
        p = accepted.p
        before = stable_parameters(p)
        beta0 = before['beta']

        def reduced(beta_array, record=False):
            nonlocal calls
            calls += 1
            beta = float(beta_array[0])
            e, d, ex, dx, eb, db, exb, dxb = completed_basis(beta)
            gram = np.array([[ip(e, ex, e, ex), ip(e, ex, d, dx)],
                             [ip(e, ex, d, dx), ip(d, dx, d, dx)]])
            load = np.array([ip(u, du, e, ex), ip(u, du, d, dx)])
            a, c = np.linalg.solve(gram, load)
            v, dv = a*e+c*d, a*ex+c*dx
            vb, dvb = a*eb+c*db, a*exb+c*dxb
            energy = ip(v-u, dv-du, v-u, dv-du)/2
            gradient = ip(v-u, dv-du, vb, dvb)/n2
            if record:
                trajectory.append({'iteration': iterations, 'evaluations': calls,
                                   'phase': 'completed', 'beta': beta, 'h': 0.0,
                                   'a': float(a), 'c': float(c), 'energy_gap': energy,
                                   'relative_error': math.sqrt(2*energy/n2),
                                   'stored_coefficient_norm': math.hypot(a,c),
                                   'stable_coefficient_norm': math.hypot(a,c),
                                   'ordinary_weight_norm': None, 'ordinary_weight_max': None})
            return energy/n2, np.array([gradient]), float(a), float(c)

        def completed_callback(beta_array):
            nonlocal iterations
            iterations += 1
            reduced(beta_array, record=True)

        reduced([beta0], record=True)
        opt = minimize(lambda q: reduced(q)[:2], [beta0], jac=True,
                       method='L-BFGS-B', callback=completed_callback,
                       options={'ftol': 1e-15, 'gtol': 1e-13, 'maxiter': 1000, 'maxls': 30})
        _, _, a, c = reduced(opt.x, record=True)
        final = {'representation': 'completed', 'beta': float(opt.x[0]), 'h': 0.0,
                 'a': a, 'c': c, 'stable_coefficient_norm': math.hypot(a,c),
                 'ordinary_weight_norm': None, 'ordinary_weight_max': None}
        completion = {'success': bool(opt.success), 'status': int(opt.status),
                      'message': str(opt.message), 'iterations': int(opt.nit),
                      'function_evaluations': int(opt.nfev)}
        stop = str(opt.message)

    with mp.workdps(80):
        checked = exact.pair_evaluation(mp.mpf(str(s)), mp.mpf(final['beta']),
                                        mp.mpf(final['h']), mp.mpf(final['a']), mp.mpf(final['c']))
        final['whole_space_80digit'] = {
            k: (mp.nstr(v, 65) if v is not None else None) for k,v in checked.items()
        }
        final['relative_error'] = float(checked['relative_error'])
        final['energy_gap'] = float(checked['energy_gap'])
        final['excess_above_centered_reference'] = float(checked['excess_above_centered_reference'])
    return {'variance': s, 'seed': seed, 'method': method, 'initial_parameters': p0.tolist(),
            'target_norm_squared_training': n2, 'iterations': iterations,
            'objective_evaluations': calls, 'wall_seconds': time.perf_counter()-start,
            'stop': stop, 'raw_optimizer_returns': raw_returns,
            'completed_optimizer_return': completion,
            'completion_count': sum(e['accepted'] for e in events),
            'scans': scans, 'events': events, 'trajectory': trajectory, 'final': final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variances', nargs='+', type=float, default=[0.99, 0.995, 0.999])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    args = parser.parse_args()
    started = time.perf_counter()
    runs = []
    references = []
    with mp.workdps(80):
        for s in args.variances:
            references.append({k: mp.nstr(v,65) for k,v in exact.reference_summary(str(s)).items()})
    for s in args.variances:
        for seed in args.seeds:
            for method in ('ordinary', 'completion'):
                row = run_one(s, seed, method)
                runs.append(row)
                final = row['final']
                print(f"s={s:g} seed={seed} {method}: E={final['relative_error']:.9g} "
                      f"h={final['h']:.4g} beta={final['beta']:.4g} "
                      f"completion={row['completion_count']} iterations={row['iterations']}", flush=True)
    result = {
        'meta': {'model': 'two Gaussian atoms with variance one in H1(R)',
                 'target': 'u_s = derivative of exp(-x^2/(2s)); s is the variance',
                 'training_quadrature': {'rule': 'trapezoid', 'interval': [-16,16], 'nodes': 3201},
                 'initialization': 'default_rng(seed): two N(0,1) weights and two U[-1.5,1.5] centers',
                 'raw_optimization': 'joint weights and centers; L-BFGS-B maxiter20000 maxfun50000 maxcor20 maxls30 ftol=gtol=0',
                 'completion': 'two consecutive flags; scans every20 accepted iterations and local return; bound tests; coefficient refit and center descent after acceptance',
                 'scope': 'Matched two-atom representation experiment; centered completed reference is analytic, not a numerical global-minimum certificate for a specified nonzero variance change.',
                 'python': platform.python_version(), 'numpy': np.__version__,
                 'scipy': scipy.__version__, 'mpmath': mp.__version__,
                 'runtime_seconds': time.perf_counter()-started,
                 'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                   for name in ('run.py','math_support.py')}},
        'variances': args.variances, 'seeds': args.seeds, 'references': references, 'runs': runs,
    }
    (HERE/'results.json').write_text(json.dumps(result, indent=1)+'\n')
    print(f"Saved {len(runs)} runs in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == '__main__':
    main()
