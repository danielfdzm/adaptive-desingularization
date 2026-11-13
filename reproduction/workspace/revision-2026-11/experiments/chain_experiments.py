#!/usr/bin/env python3
"""Higher-order recovery with sequential collinear completion (revision, Nov 2026).

Problems (lambda = 1, Poisson--Boltzmann energy in the gap form of Appendix C):
  second1d  d = 1, sigma = 0.65, u* = 0.4 G''(. - 0.23), P = 3; independent start: atoms at c-h, c, c+h with
            c ~ U[-0.15, 0.15], h ~ U[0.85, 1.15]; 384 Gauss--Legendre nodes on [-12, 12].
  higher    d >= 5, sigma = 0.45, the six centres and directions of the first-order family; centres 1-2 carry
            sigma^2 b_j d_e^2 G, centres 3-4 carry sigma^3 b_j d_e^3 G, all six carry a_j G; P = 16.
            Near-target start: r+1 atoms on a perturbed axis through a perturbed centre (spacing 0.05-0.10),
            one atom near each singleton centre; target-centred scrambled Sobol rule (6 * 2^12 points).
Methods: standard (S), stable (NC: pairs in stable coordinates, no completion), adaptive (C: sequential
collinear completion).  Usage:
  python3 chain_experiments.py --problem second1d --seeds 40 50 --jobs 6 --output results/chain-second1d
  python3 chain_experiments.py --problem higher --dimensions 5 --seeds 10 15 --jobs 6 --output results/chain-higher
"""
from __future__ import annotations

import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

HERE = Path(__file__).resolve().parent
PB = HERE.parents[1] / "adaptive-desingularization" / "final-experiments" / "pb"
sys.path.insert(0, str(PB)); sys.path.insert(0, str(HERE))
import chain_solver as cs                                   # noqa: E402
import config as pbconfig                                   # noqa: E402  (first-order geometry, unchanged)
from quadrature import make_quadrature                      # noqa: E402

A_COEF = np.array([1.1, -0.8, 1.4, -1.2, 0.9, -0.7])
B_COEF = np.array([-1.3, 1.5, 0.9, -1.1])
ORDERS = [2, 2, 3, 3]


def problem_second1d(seed):
    sigma = 0.65
    target = (np.array([[0.23]]), np.array([[1.0]]), np.array([2]), np.array([0.4]))
    rng = np.random.default_rng(seed)
    c, h = rng.uniform(-0.15, 0.15), rng.uniform(0.85, 1.15)
    starts = [np.array([c - h]), np.array([c]), np.array([c + h])]
    x, w = roots_legendre(384)
    X, W = 12.0 * x[:, None], 12.0 * w
    return dict(d=1, sigma=sigma, target=target, starts=starts, X=X, W=W, box=3.0, max_fits=1000)


def problem_higher(d, geometry, seed):
    sigma = 0.45
    c5, e5 = pbconfig.base_geometry(geometry)
    q = pbconfig.embedding(d, geometry)
    cen, dirs = c5 @ q.T, e5 @ q.T
    tc, tu, to, tw = [], [], [], []
    unit = np.zeros(d); unit[0] = 1.0                       # order-0 primitives need no direction
    for j in range(6):
        tc.append(cen[j]); tu.append(dirs[j] if j < 4 else unit); to.append(0); tw.append(A_COEF[j])
    for j in range(4):
        r = ORDERS[j]
        tc.append(cen[j]); tu.append(dirs[j]); to.append(r); tw.append(sigma ** r * B_COEF[j])
    target = (np.array(tc), np.array(tu), np.array(to), np.array(tw))
    rng = np.random.default_rng(100000 * geometry + seed)
    starts = []
    for j in range(6):
        perturb = rng.normal(size=d); perturb *= rng.uniform(.06, .14) / np.linalg.norm(perturb)
        centre = cen[j] + perturb
        if j < 4:
            tangent = rng.normal(size=d); tangent -= tangent @ dirs[j] * dirs[j]; tangent /= np.linalg.norm(tangent)
            angle = rng.uniform(-.15, .15)
            axis = np.cos(angle) * dirs[j] + np.sin(angle) * tangent
            r = ORDERS[j]; spacing = rng.uniform(.05, .10)
            for i in range(r + 1):
                starts.append(centre + (i - r / 2) * spacing * axis)
        else:
            starts.append(centre)
    X, W, _ = make_quadrature(cen, sigma=sigma, power=12, seed=8128 + 100 * geometry + d)
    return dict(d=d, sigma=sigma, target=target, starts=starts, X=X, W=W, box=2.0, max_fits=3000)


def run_one(task):
    problem, d, geometry, seed, method = task
    P = problem_second1d(seed) if problem == "second1d" else problem_higher(d, geometry, seed)
    solver = cs.Solver(P["d"], P["sigma"], P["target"], P["starts"], method, P["X"], P["W"], lam=1.0,
                       max_fits=P["max_fits"], box=P["box"])
    out = solver.run()
    out.update(problem=problem, geometry=geometry, seed=seed, slots=len(P["starts"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", choices=["second1d", "higher"], required=True)
    ap.add_argument("--dimensions", nargs="+", type=int, default=[5])
    ap.add_argument("--geometries", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--seeds", nargs=2, type=int, required=True, help="half-open interval")
    ap.add_argument("--methods", nargs="+", default=["standard", "stable", "adaptive"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.problem == "second1d":
        tasks = [("second1d", 1, 0, s, m) for s in range(*args.seeds) for m in args.methods]
    else:
        tasks = [("higher", d, g, s, m) for d in args.dimensions for g in args.geometries
                 for s in range(*args.seeds) for m in args.methods]
    name = lambda t: f"{t[0]}_d{t[1]}_g{t[2]}_seed{t[3]}_{t[4]}"
    srcs = [HERE / "chain_solver.py", HERE / "chain_kernels.py", HERE / "chain_experiments.py",
            PB / "fit.py", PB / "quadrature.py", PB / "config.py"]
    (args.output / "provenance.json").write_text(json.dumps({
        "command": sys.argv, "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}}, indent=1))
    todo = [t for t in tasks if not (args.output / f"{name(t)}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, t): t for t in todo}
        for f in as_completed(futures):
            t = futures[f]
            try:
                r = f.result()
            except Exception as exc:
                (args.output / f"{name(t)}.error.txt").write_text(repr(exc)); print(name(t), "FAILED", repr(exc), flush=True)
                continue
            (args.output / f"{name(t)}.json").write_text(json.dumps(r, indent=1, default=float))
            print(f"{name(t)} error={r['final_relative_error']:.3e} fits={r['fits']} completions={r['completions']} "
                  f"joins={r['joins']} blocks={r['blocks_final']} stop={r['stop']} wall={r['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
