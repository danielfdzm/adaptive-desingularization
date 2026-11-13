#!/usr/bin/env python3
"""S reruns with the two-return stopping rule for the two Table-2 studies whose S runs stopped at the
first local return: linear d=1 (seeds 0-9) and PB d=1, r=2 (seeds 40-49). All other inputs unchanged.

Linear d=1: StandardSolver1D with a derived run loop (only change: stop after two optimizer returns
instead of one; status-1 returns are not counted, as before). The frozen loop is also rerun to check
that it reproduces the historical S records, and the derived loop's first return is checked against them.
PB d=1, r=2: the guarded chain solver's existing 'standard_restart' method (two returns).
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AD = ROOT / 'adaptive-desingularization'
GUARDED = ROOT / 'numerical-section-revision' / 'guarded-rerun-2026-11-09'
sys.path.insert(0, str(AD / 'experiments' / 'core'))
sys.path.insert(0, str(AD / 'experiments' / 'manufactured_1d'))
import numpy as np
from scipy.optimize import minimize
import run as lin                                  # experiments/manufactured_1d/run.py (unchanged)
from adaptive1d import StandardSolver1D


class StandardTwoReturns(StandardSolver1D):
    def run(self):
        t0 = time.time()
        w, M, g, cond = self.fit(self.centers)
        if w is None:
            self.stop_reason = "ill-conditioned at start"; self.normal_return = False; self.record()
            self.wall = time.time() - t0
            return
        self.weights = w
        self.record()
        bounds = [(-self.prob.L0, self.prob.L0)] * self.P
        self._nrec = 0
        self.first_return = None
        stalls = 0

        def cb(xk):
            self._nrec += 1
            if self._nrec % self.scan_period == 0:
                w, M, g, cond = self.fit(xk, strict=False)
                if w is not None and np.all(np.isfinite(w)):
                    self.centers, self.weights = xk.copy(), w
                    self.record()
            if self.n_fits >= self.max_fits:
                raise StopIteration
        while self.n_fits < self.max_fits:
            try:
                res = minimize(self.objective, self.centers, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                               options={"maxiter": 100000, "ftol": self.ftol, "gtol": self.gtol, "maxls": 40})
            except StopIteration:
                self.stop_reason = "work_limit"
                w, M, g, cond = self.fit(self.centers)
                if w is not None:
                    self.weights = w
                self.record()
                break
            except FloatingPointError:
                self.stop_reason = "ill-conditioned"; self.normal_return = False
                break
            self.lbfgs_iters += res.nit
            w, M, g, cond = self.fit(res.x)
            if w is None:
                self.stop_reason = "ill-conditioned"; self.normal_return = False
                break
            self.centers, self.weights = res.x.copy(), w
            self.record()
            if res.status in (0, 2):                                  # CHANGED: count returns, stop after two
                if self.first_return is None:
                    self.first_return = dict(fits=self.n_fits, rel_err=self.trace.rel_err[-1], status=int(res.status),
                                             centers=self.centers.tolist())
                if res.status == 2:
                    self.normal_return = False
                stalls += 1
                if stalls >= 2:
                    self.stop_reason = "converged" if res.status == 0 else "abnormal: " + str(res.message)
                    break
        if self.stop_reason is None:
            self.stop_reason = "work_limit"
        self.wall = time.time() - t0


def linear():
    hist = {r['seed']: r for r in json.loads((GUARDED / 'historical-baselines' / 'linear1d-shared.json').read_text())['runs']
            if r['kind'] == 'standard'}
    rows = []
    for seed in range(10):
        prob, _ = lin.make_problem(None)
        centers = lin.near_target_start(seed)
        frozen = StandardSolver1D(prob, centers, max_fits=3000); frozen.run()
        fr = lin.summarize_standard(frozen)
        prob, _ = lin.make_problem(None)
        two = StandardTwoReturns(prob, centers, max_fits=3000); two.run()
        tr = lin.summarize_standard(two)
        h = hist[seed]
        rows.append(dict(study='linear d=1', seed=seed,
                         frozen_reproduces_history=(fr['final_rel_err'] == h['final_rel_err'] and fr['fits'] == h['fits']),
                         first_return_matches_history=(two.first_return is not None and two.first_return['fits'] == h['fits']
                                                       and two.first_return['rel_err'] == h['final_rel_err']),
                         historical_error=h['final_rel_err'], historical_fits=h['fits'], historical_stop=h['stop'],
                         error=tr['final_rel_err'], fits=tr['fits'], stop=tr['stop'], record=tr))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != 'record'}), flush=True)
    return rows


def second1d():
    sys.path.insert(0, str(GUARDED / 'directional' / 'src'))
    import chain_experiments as ce
    rows = []
    for seed in range(40, 50):
        h = json.loads((GUARDED / 'historical-baselines' / f'second1d_d1_g0_seed{seed}_standard.json').read_text())
        r = ce.run_one(('second1d', 1, 0, seed, 'standard_restart'))
        first = next((t for t in r['trace'] if t['fits'] == h['fits']), None)
        rows.append(dict(study='PB d=1 r=2', seed=seed, historical_error=h['final_relative_error'], historical_fits=h['fits'],
                         historical_stop=h['stop'],
                         first_return_matches_history=first is not None and abs(first['relative_error'] - h['trace'][-1]['relative_error']) <= 1e-15,
                         error=r['final_relative_error'], fits=r['fits'], stop=r['stop'], record=r))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != 'record'}), flush=True)
    return rows


if __name__ == '__main__':
    out = dict(linear=linear(), second1d=second1d(),
               sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), AD / 'experiments/core/adaptive1d.py', AD / 'experiments/manufactured_1d/run.py',
                         GUARDED / 'directional/src/chain_solver.py', GUARDED / 'directional/src/chain_experiments.py']})
    (HERE / 'results.json').write_text(json.dumps(out, default=float) + '\n')
