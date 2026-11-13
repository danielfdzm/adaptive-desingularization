#!/usr/bin/env python3
"""One-dimensional manufactured problems for the adaptive algorithm (Section 3.2--3.3).

Studies (``--studies``):
  shared       ten near-target starts, capacity N=6: standard, adaptive, stable-coordinates ablation;
  generic      equispaced starts without prescribed support, N in {6, 8, 10};
Every run records its trace, events and final state in JSON. Coefficients are unconstrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[name]="1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "core"))
from gauss1d import AtomTable, Block, Kernel1D
from adaptive1d import (AdaptiveConfig, AdaptiveSolver1D, Problem1D, StandardSolver1D, default_rho_fn)

SIGMA = 0.6
TARGET = dict(centers=[0.0, 0.0, 2.2, 2.2, 2.2, -2.1], orders=[0, 1, 0, 1, 2, 0], coeffs=[0.8, -1.2, -0.9, 0.6, 0.5, 1.0])
C_FACTOR = None
MAX_FITS = 3000
SOURCES = ["experiments/manufactured_1d/run.py", "experiments/core/gauss1d.py", "experiments/core/coefficient_fit.py",
           "experiments/core/adaptive1d.py"]


def make_problem(c_factor=None):
    kern = Kernel1D(SIGMA)
    target = AtomTable(np.array(TARGET["centers"]), np.array(TARGET["orders"]), np.array(TARGET["coeffs"]))
    rho_fn = default_rho_fn(kern)
    return Problem1D(kern,target,rho_fn),None


def near_target_start(seed):
    rng = np.random.default_rng(seed)
    c1 = -2.1 + rng.uniform(-0.15, 0.15)
    c2 = rng.uniform(-0.15, 0.15); h2 = rng.uniform(0.06, 0.15)
    c3 = 2.2 + rng.uniform(-0.15, 0.15); s3 = rng.uniform(0.08, 0.18)
    return np.array([c1, c2 - h2, c2 + h2, c3 - s3 * (1 + rng.uniform(-0.3, 0.3)), c3, c3 + s3 * (1 + rng.uniform(-0.3, 0.3))])


def generic_start(seed, N):
    rng = np.random.default_rng(100 + seed)
    return np.linspace(-3, 3, N) + rng.uniform(-0.3, 0.3, N)


def tofloat(x):
    if isinstance(x, (np.floating, np.integer)):
        return tofloat(x.item())
    if isinstance(x, np.ndarray):
        return tofloat(x.tolist())
    if isinstance(x, dict):
        return {k: tofloat(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tofloat(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):
        return None
    return x


def summarize_adaptive(sol, cert):
    tr = sol.trace
    return tofloat({
        "final_rel_err": tr.rel_err[-1], "final_rel_err_quad": tr.rel_err_quad[-1], "best_rel_err": sol.best_rel,
        "fits": sol.n_fits, "wall": sol.wall, "coef_norm": tr.coef_norm[-1], "raw_norm": tr.raw_norm[-1],
        "max_raw_norm": max(tr.raw_norm), "min_halfgap": tr.min_halfgap[-1], "cost": tr.cost[-1], "C": sol.C,
        "slots": sol.slots, "completions": sol.n_completions, "insertions": sol.n_insertions,
        "normal_return": sol.normal_return, "abnormal_returns": sol.abnormal_returns, "stop": sol.stop_reason,
        "global_certificate": False, "coefficient_constraint": None, "excess": cert["excess"], "transfer_rejections": sol.transfer_rejections,
        "unresolved_flags": sol.unresolved_flags, "lbfgs_iters": sol.lbfgs_iters,
        "blocks": [{"nodes": b.nodes, "mult": b.mult, "gamma": b.gamma} for b in sol.blocks],
        "trace": {"fits": tr.fits, "rel_err": tr.rel_err, "rel_err_quad": tr.rel_err_quad, "coef_norm": tr.coef_norm,
                  "raw_norm": tr.raw_norm, "min_halfgap": tr.min_halfgap, "cost": tr.cost, "slots": tr.slots, "energy": tr.energy},
        "events": [{"fits": f, "kind": k, **d} for f, k, d in tr.events],
        "gaps": [{k: v for k, v in g.items() if k != "candidates"} for g in tr.gaps],
    })


def summarize_standard(std):
    tr = std.trace
    return tofloat({
        "final_rel_err": tr.rel_err[-1], "final_rel_err_quad": tr.rel_err_quad[-1], "best_rel_err": std.best_rel,
        "fits": std.n_fits, "wall": std.wall, "coef_norm": tr.coef_norm[-1], "raw_norm": tr.raw_norm[-1],
        "max_raw_norm": max(tr.raw_norm), "min_halfgap": tr.min_halfgap[-1], "normal_return": std.normal_return,
        "C": None, "coefficient_constraint": None, "stop": std.stop_reason, "max_cond": std.max_cond, "lbfgs_iters": std.lbfgs_iters,
        "centers": std.centers, "weights": std.weights,
        "trace": {"fits": tr.fits, "rel_err": tr.rel_err, "rel_err_quad": tr.rel_err_quad, "coef_norm": tr.coef_norm,
                  "min_halfgap": tr.min_halfgap, "energy": tr.energy},
    })


def run_one(args):
    kind, seed, N, c_factor, start_kind, max_fits = args
    prob, R = make_problem(c_factor)
    centers = near_target_start(seed) if start_kind == "near" else generic_start(seed, N)
    if kind == "standard":
        std = StandardSolver1D(prob, centers, max_fits=max_fits)
        std.run()
        return summarize_standard(std)
    cfg = AdaptiveConfig(N=N, max_fits=max_fits, completion=(kind == "adaptive"))
    sol = AdaptiveSolver1D(prob, [Block([c], [0.0]) for c in centers], cfg)
    cert = sol.run()
    return summarize_adaptive(sol, cert)


def study_shared(pool, seeds, max_fits):
    jobs = [(kind, s, 6, C_FACTOR, "near", max_fits) for s in seeds for kind in ("standard", "adaptive", "ablation")]
    res = pool.map(run_one, jobs)
    out = {"meta": {"sigma": SIGMA, "target": TARGET, "C_factor": C_FACTOR, "N": 6, "seeds": list(seeds), "max_fits": max_fits}, "runs": []}
    for (kind, s, *_), r in zip(jobs, res):
        r["kind"] = kind; r["seed"] = s
        out["runs"].append(r)
    return out


def study_generic(pool, seeds, max_fits, Ns=(6, 8, 10)):
    jobs = [(kind, s, N, C_FACTOR, "generic", max_fits) for N in Ns for s in seeds for kind in ("standard", "adaptive", "ablation")]
    res = pool.map(run_one, jobs)
    out = {"meta": {"sigma": SIGMA, "target": TARGET, "C_factor": C_FACTOR, "Ns": list(Ns), "seeds": list(seeds), "max_fits": max_fits}, "runs": []}
    for (kind, s, N, *_), r in zip(jobs, res):
        r["kind"] = kind; r["seed"] = s; r["N"] = N
        out["runs"].append(r)
    return out


def provenance(command, elapsed, extra=None):
    import scipy
    info = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": command,
        "elapsed_seconds": round(elapsed, 1),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
    }
    if extra:
        info.update(extra)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/unconstrained-manufactured-1d")
    parser.add_argument("--studies", nargs="+", default=["shared", "generic"],
                        choices=["shared", "generic"])
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 10), metavar=("FIRST", "LAST"), help="seed range [FIRST, LAST)")
    parser.add_argument("--max-fits", type=int, default=MAX_FITS)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 2:
        parser.error("use one or two workers")
    if args.output.exists():
        parser.error(f"output directory exists: {args.output}")
    args.output.mkdir(parents=True)
    seeds = range(args.seeds[0], args.seeds[1])
    command = " ".join(sys.argv)
    files = {"shared": "shared_starts.json", "generic": "generic_starts.json"}
    timings = {}
    with Pool(args.jobs) as pool:
        for study in args.studies:
            t0 = time.time()
            if study == "shared":
                out = study_shared(pool, seeds, args.max_fits)
            elif study == "generic":
                out = study_generic(pool, seeds, args.max_fits)
            (args.output / files[study]).write_text(json.dumps(out, allow_nan=False) + "\n")
            timings[study] = round(time.time() - t0, 1)
            print(f"{study}: {timings[study]} s", flush=True)
    (args.output / "provenance.json").write_text(json.dumps(provenance(command, sum(timings.values()), {"study_seconds": timings, "coefficient_constraint": None, "threads_per_worker": 1}), indent=2) + "\n")


if __name__ == "__main__":
    main()
