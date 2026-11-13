#!/usr/bin/env python3
"""Two-dimensional tests with independent lattice initialization and a target exactly in the completed class
(perturbation 0): dipole and tilted, P=4, 64^2 Gauss-Legendre nodes, 1000 fits, seeds 101-105.
Methods: standard (S), stable (NC), adaptive (C) from pb-independent-2d/run.py (loaded by path, unmodified),
and closure (T): NC with pair half-gaps optimized through t=h^2 (see pb_variants.py).
Usage: independent2d_variants.py OUTPUT_DIR JOBS"""
import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"
import importlib.util, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = HERE.parents[1] / "adaptive-desingularization/final-experiments/pb-independent-2d"
sys.path.insert(0, str(D)); sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ind2d", D / "run.py")
ind2d = importlib.util.module_from_spec(spec); spec.loader.exec_module(ind2d)
from pb_variants import closure_geometry
BASE_GEOMETRY = ind2d.base.Geometry
CLOSURE_GEOMETRY = closure_geometry(ind2d.base)

def run(task):
    case, seed, method = task
    ind2d.base.Geometry = CLOSURE_GEOMETRY if method == "closure" else BASE_GEOMETRY
    m = "stable" if method == "closure" else method
    r = ind2d.IndependentSolver(case, seed, m, 1000, 64, 0.0, 4, "lattice").run()
    r["variant"] = method
    return r

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    os.environ["PB_PROGRESS_DIR"] = str((out / "progress").resolve()); (out / "progress").mkdir(exist_ok=True)
    tasks = [(c, s, m) for c in ("dipole", "tilted") for s in range(101, 106) for m in ("standard", "stable", "adaptive", "closure")
             if not (out / f"{c}_{s}_{m}.json").exists()]
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as pool:
        futs = {pool.submit(run, t): t for t in tasks}
        for f in as_completed(futs):
            c, s, m = futs[f]
            try:
                r = f.result()
            except Exception as exc:
                (out / f"{c}_{s}_{m}.error.txt").write_text(repr(exc)); print(c, s, m, "FAILED", repr(exc), flush=True); continue
            (out / f"{c}_{s}_{m}.json").write_text(json.dumps(r, indent=1, default=str))
            print(c, s, m, "error %.3e" % r["final_relative_error"], "fits", r["fits"], "completions", r["completions"], "stop", r["stop"], flush=True)
