#!/usr/bin/env python3
"""Pre-registered one-dimensional independent recovery cohort (Table 1, panel A).

Orders 2 and 3, seeds 900-999, methods S (standard_restart), NC (stable) and
C (adaptive) of the unchanged guarded directional solver. Case generation,
solver call and endpoint validation are those of
`guarded-rerun-2026-11-09/independent/run.py`; S is added with the same
two-return stopping rule. Existing outputs are kept.
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
ROOT = HERE.parents[2]
GUARDED = ROOT / "numerical-section-revision" / "guarded-rerun-2026-11-09"
SRC = GUARDED / "directional" / "src"
sys.path.insert(0, str(SRC))
import chain_solver as cs          # noqa: E402
import chain_kernels as ck         # noqa: E402
from quadrature import bregman_parts  # noqa: E402

METHODS = {"S": "standard_restart", "NC": "stable", "C": "adaptive"}


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
    order, seed, label = task
    sigma, shift, factor, target, starts = case(order, seed)
    X, W = rule(384, 10.)
    t0 = time.perf_counter()
    solver = cs.Solver(1, sigma, target, starts, METHODS[label], X, W, lam=10., max_fits=1200, box=3.)
    setup_seconds = time.perf_counter() - t0
    result = solver.run()
    result.update(order=order, seed=seed, phase3_method=label, sigma=sigma, lambda_value=10.,
                  target=[x.tolist() for x in target], target_shift=shift, target_amplitude_factor=factor,
                  initial_centers=starts.tolist(), width=order + 1, stage="phase-3 enlarged cohort",
                  setup_seconds=setup_seconds)
    result["end_to_end_seconds"] = setup_seconds + result["solve_seconds"]
    start_check = time.perf_counter()
    try:
        result["validation"] = validate(solver, result)
    except Exception as exc:          # ordinary S endpoints can be singular; keep the run, record the failure
        result["validation"] = dict(error=repr(exc))
    result["independent_validation_seconds"] = time.perf_counter() - start_check
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(900, 1000))
    sources = [Path(__file__)] + sorted(SRC.glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(seeds=seeds, orders=[2, 3], methods=METHODS, workers=args.jobs, source_sha256=hashes,
                    prereg_sha256=(ROOT / "work/sinum-phase3-2026-11-11/prereg/prereg.sha256").read_text(),
                    numpy=np.__version__, torch=torch.__version__, python=platform.python_version(),
                    stopping="two local returns without structural change, or 1200 fits, for every method",
                    timing="accuracy campaign with several workers; wall times not used for comparisons")
    manifest = out / "protocol.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["source_sha256"] != hashes or old["seeds"] != seeds:
            raise RuntimeError("Protocol differs; use a new directory")
    else:
        write(manifest, protocol)
    tasks = [(r, s, m) for r in (2, 3) for s in seeds for m in ("S", "NC", "C")]
    name = lambda t: f"order{t[0]}_seed{t[1]}_{t[2]}.json"
    todo = [t for t in tasks if not (out / name(t)).exists()]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, t): t for t in todo}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(order=task[0], seed=task[1], phase3_method=task[2], stop="exception", exception=repr(exc))
            write(out / name(task), result)
            print(name(task), result.get("final_relative_error"), result.get("fits"), result.get("stop"), flush=True)
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during campaign: {p}")
    write(out / "complete.json", dict(count=len(tasks)))


if __name__ == "__main__":
    main()
