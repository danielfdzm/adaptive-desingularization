#!/usr/bin/env python3
"""Manufactured higher-order 2D recovery from independent lattice starts.

Calls the existing chain solver without modifying its source or behavior.
Only pilot seeds 200--202 are authorized by this script's default protocol.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
from scipy.special import roots_legendre
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = ROOT / "revision-2026-11/experiments"
PB = ROOT / "adaptive-desingularization/final-experiments/pb"
sys.path.insert(0, str(EXP))
import chain_solver as cs
import chain_kernels as ck
from quadrature import bregman_parts


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def rule(n, box=6.0):
    x, w = roots_legendre(n)
    x, w = box * x, box * w
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]), np.outer(w, w).ravel()


def problem(order, seed, nodes=96, lam=1.0, max_fits=1200):
    sigma = 0.65
    angle = 0.73 if order == 2 else 1.17
    centre = np.array([0.31, -0.23]) if order == 2 else np.array([-0.28, 0.37])
    direction = np.array([np.cos(angle), np.sin(angle)])
    target = (centre[None, :], direction[None, :], np.array([order]), np.array([sigma ** order]))
    lattice = (np.array([[-0.9, -0.6], [0.0, 0.6], [0.9, -0.6]]) if order == 2
               else np.array([[-0.85, -0.85], [-0.85, 0.85], [0.85, -0.85], [0.85, 0.85]]))
    starts = lattice + np.random.default_rng(seed).uniform(-0.12, 0.12, lattice.shape)
    X, W = rule(nodes)
    return dict(d=2, sigma=sigma, target=target, starts=starts, X=X, W=W,
                lam=lam, max_fits=max_fits, box=3.0)


def values(X, blocks, coef, sigma):
    C, U, O, WS = cs.primitives(blocks, 2)
    pw = WS @ torch.as_tensor(coef)
    return np.concatenate([(ck.point_values(torch.as_tensor(X[i:i + 4096]), C, U, O, sigma) @ pw).numpy()
                           for i in range(0, len(X), 4096)])


def validate(solver, run, nodes):
    target2 = float(ck.mp_norm2(solver.target_np, solver.sigma, 2, dps=80))
    quadratic_gap = 0.5 * run["final_relative_error"] ** 2 * target2
    rows = []
    for n, box in ((nodes, 6.0), (2 * nodes, 6.0), (2 * nodes, 8.0)):
        X, W = rule(n, box)
        u = (ck.point_values(torch.as_tensor(X), *solver.target[:3], solver.sigma) @ solver.target[3]).numpy()
        v = values(X, solver.blocks, solver.coef, solver.sigma)
        gap, _, _ = bregman_parts(v, u)
        Fu, _, _ = bregman_parts(u, np.zeros_like(u))
        nonlinear_gap = float(solver.lam_record * (W @ gap))
        rows.append(dict(nodes_per_axis=n, box=box, quadratic_gap=quadratic_gap,
                         nonlinear_gap=nonlinear_gap, total_gap=quadratic_gap + nonlinear_gap,
                         target_nonlinear_over_quadratic=float(solver.lam_record * (W @ Fu) / (.5 * target2)),
                         target_peak_sampled=float(np.max(np.abs(u)))))
    blocks = [{k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in b.items()}
              for b in solver.blocks]
    M, b, Phi = (x.detach().numpy() for x in cs.assemble(blocks, 2, solver.sigma, solver.X, solver.target))
    _, residual, _ = bregman_parts(Phi @ solver.coef, solver.Uq)
    grad = M @ solver.coef - b + Phi.T @ (solver.Wq * residual)
    scaled_kkt = np.max(np.abs(grad) / np.sqrt(np.diag(M))) / math.sqrt(target2)
    return dict(quadrature=rows, normalized_coefficient_kkt=float(scaled_kkt),
                maximum_relative_gap_change=max(abs(row["total_gap"] / rows[-1]["total_gap"] - 1)
                                               for row in rows[:-1]) if rows[-1]["total_gap"] > 0 else 0.0)


def run_one(task):
    order, seed, method, nodes, lam, max_fits = task
    P = problem(order, seed, nodes, lam, max_fits)
    solver = cs.Solver(P["d"], P["sigma"], P["target"], P["starts"], method, P["X"], P["W"],
                       lam=P["lam"], max_fits=P["max_fits"], box=P["box"])
    solver.lam_record = lam
    run = solver.run()
    run.update(case=f"blind_order{order}", order=order, seed=seed, lambda_value=lam,
               sigma=P["sigma"], target=[a.tolist() for a in P["target"]],
               initial_centers=P["starts"].tolist(), width=order + 1,
               independent_initialization=True, fixed_target_independent_quadrature=True,
               manufactured_target=True)
    run["validation"] = validate(solver, run, nodes)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202])
    parser.add_argument("--methods", nargs="+", default=["standard_restart", "stable", "adaptive"])
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--max-fits", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=HERE / "pilot")
    args = parser.parse_args()
    if any(seed >= 900 for seed in args.seeds):
        parser.error("Holdout seeds are intentionally disabled in this pilot script.")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), EXP / "chain_solver.py", EXP / "chain_kernels.py", PB / "fit.py", PB / "quadrature.py"]
    protocol = dict(purpose="manufactured higher-order recovery from target-independent lattice starts",
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                    source_behavior="unmodified chain_solver; endpoint validation added externally",
                    target_is_known_only_for_manufactured_training_objective=True,
                    fixed_cases=[dict(order=r, target=[a.tolist() for a in problem(r, 200)["target"]]) for r in args.orders])
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != clean(protocol):
        raise RuntimeError("Protocol mismatch: use a new output directory for changed settings or sources")
    save(protocol_path, protocol)
    tasks = [(r, s, m, args.nodes, args.lam, args.max_fits) for r in args.orders for s in args.seeds for m in args.methods]
    name = lambda t: f"order{t[0]}_seed{t[1]}_{t[2]}"
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks if not (args.output / f"{name(task)}.json").exists()}
        for future in as_completed(futures):
            task = futures[future]
            try:
                run = future.result()
            except Exception as exc:
                run = dict(case=f"blind_order{task[0]}", order=task[0], seed=task[1], method=task[2],
                           stop="exception", exception=repr(exc))
            save(args.output / f"{name(task)}.json", run)
            print(name(task), "error", run.get("final_relative_error"), "fits", run.get("fits"),
                  "completed", run.get("completions"), "joined", run.get("joins"), "stop", run["stop"], flush=True)


if __name__ == "__main__":
    main()
