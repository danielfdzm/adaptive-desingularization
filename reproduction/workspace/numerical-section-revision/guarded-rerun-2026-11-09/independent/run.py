#!/usr/bin/env python3
"""Repeat the frozen independent recovery cohort with the published guards.

Optimization is timed separately from independent endpoint validation. Accuracy
runs can use two workers; --benchmark uses one process and the predeclared seeds.
The S baseline is unchanged and remains in the historical confirmation archive.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import roots_legendre
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "directional" / "src"
ROOT = HERE.parents[2]
sys.path.insert(0, str(SRC))
import chain_solver as cs
import chain_kernels as ck
from quadrature import bregman_parts


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rule(nodes, box):
    x, w = roots_legendre(nodes)
    return box * x[:, None], box * w


def case(order, seed):
    sigma = .65
    rng = np.random.default_rng(100000 + seed)
    shift, factor = float(rng.uniform(-.6, .6)), float(rng.uniform(.7, 1.3))
    target = (np.array([[shift]]), np.array([[1.]]), np.array([order]),
              np.array([factor * sigma**order]))
    starts = np.random.default_rng(seed).uniform(-1.2, 1.2, (order + 1, 1))
    return sigma, shift, factor, target, starts


def validate(solver, result):
    target2 = float(ck.mp_norm2(solver.target_np, solver.sigma, 1, dps=80))
    quadratic = .5 * result["final_relative_error"]**2 * target2
    C, U, O, WS = cs.primitives(solver.blocks, 1, solver.sigma)
    weights = WS @ torch.as_tensor(solver.coef)
    rows = []
    for nodes, box in ((384, 10.), (768, 10.), (768, 12.)):
        X, W = rule(nodes, box)
        xt = torch.as_tensor(X)
        u = (ck.point_values(xt, *solver.target[:3], solver.sigma) @ solver.target[3]).numpy()
        v = (ck.point_values(xt, C, U, O, solver.sigma) @ weights).numpy()
        gap, _, _ = bregman_parts(v, u)
        potential, _, _ = bregman_parts(u, np.zeros_like(u))
        nonlinear = float(10. * (W @ gap))
        rows.append(dict(nodes=nodes, box=box, quadratic_gap=quadratic,
                         nonlinear_gap=nonlinear, total_gap=quadratic + nonlinear,
                         target_nonlinear_over_quadratic=float(10. * (W @ potential) / (.5 * target2))))
    blocks = [{k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
               for k, v in b.items()} for b in solver.blocks]
    M, b, Phi = (x.detach().numpy() for x in cs.assemble(blocks, 1, solver.sigma, solver.X, solver.target))
    _, residual, _ = bregman_parts(Phi @ solver.coef, solver.Uq)
    gradient = M @ solver.coef - b + Phi.T @ (solver.Wq * residual)
    return dict(quadrature=rows,
                normalized_coefficient_kkt=float(np.max(np.abs(gradient) / np.sqrt(np.diag(M))) / np.sqrt(target2)),
                normalized_maximum_gap_change=max(abs(x["total_gap"] - rows[-1]["total_gap"]) / target2
                                                 for x in rows[:-1]))


def run_one(task):
    order, seed, method, benchmark = task
    sigma, shift, factor, target, starts = case(order, seed)
    X, W = rule(384, 10.)
    t0 = time.perf_counter()
    solver = cs.Solver(1, sigma, target, starts, method, X, W,
                       lam=10., max_fits=1200, box=3.)
    setup_seconds = time.perf_counter() - t0
    result = solver.run()
    result.update(order=order, seed=seed, sigma=sigma, lambda_value=10.,
                  target=[x.tolist() for x in target], target_shift=shift,
                  target_amplitude_factor=factor, initial_centers=starts.tolist(),
                  width=order + 1, stage="guarded confirmation rerun", benchmark=benchmark,
                  setup_seconds=setup_seconds,
                  success_at_1e_6=result["final_relative_error"] <= 1e-6,
                  high_accuracy_at_1e_10=result["final_relative_error"] <= 1e-10)
    result["end_to_end_seconds"] = setup_seconds + result["solve_seconds"]
    start_check = time.perf_counter()
    result["validation"] = validate(solver, result)
    result["independent_validation_seconds"] = time.perf_counter() - start_check
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args()
    seeds = args.seeds or (list(range(900, 903)) if args.benchmark else list(range(900, 910)))
    workers = 1 if args.benchmark else args.jobs
    out = HERE / ("benchmark" if args.benchmark else "accuracy")
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__)] + sorted(SRC.glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(seeds=seeds, orders=[2, 3], methods=["stable", "adaptive"], workers=workers,
                    benchmark=args.benchmark, source_sha256=hashes,
                    numpy=np.__version__, torch=torch.__version__, python=platform.python_version(),
                    timing_scope="Solver construction, initial fitting, optimizer, detection, trials, completion and restarts; excludes problem/quadrature generation and posthoc high-precision checks",
                    matched_original="work/independent-tests-2026-11-08/blind_higher_order/holdout",
                    changed_rule="Equation C.4, dimensionless transverse storage, resolved new directions")
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise RuntimeError("Protocol differs; preserve these records and use a new directory")
    write(manifest, protocol)
    tasks = [(r, s, m, args.benchmark) for r in (2, 3) for s in seeds
             for m in (("stable", "adaptive") if s % 2 == 0 else ("adaptive", "stable"))]
    name = lambda t: f"order{t[0]}_seed{t[1]}_{t[2]}.json"
    todo = [t for t in tasks if not (out / name(t)).exists()]
    if workers == 1:
        for task in todo:
            result = run_one(task)
            write(out / name(task), result)
            print(name(task), result["final_relative_error"], result["fits"], result["end_to_end_seconds"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, t): t for t in todo}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                write(out / name(task), result)
                print(name(task), result["final_relative_error"], result["fits"], result["end_to_end_seconds"], flush=True)
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during campaign: {p}")


if __name__ == "__main__":
    main()
