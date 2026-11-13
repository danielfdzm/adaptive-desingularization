#!/usr/bin/env python3
"""Additional comparison methods for the Poisson--Boltzmann recovery suites (revision, Nov 2026).

Variants
  closure   NC (method 'stable') with every ordinary pair's half-gap optimized through t = h^2 on
            the same feasible set as NC, t in [(1e-12 sigma)^2, (0.6 sigma)^2].  No completion step:
            the optimizer works on a parametrization that extends smoothly to the collided pair.
  srestart  S (method 'standard') with the restart rule of NC/C: stop after two consecutive local
            optimizer returns without a structural change.  S never changes structure, so this is S
            with one restart from its first return.

The solver sources in adaptive-desingularization/final-experiments/{pb,higher-order} are loaded by
path and are not modified; the restart rule is obtained by one checked substitution in a copy of
Solver.run, and the closure coordinates by subclassing Geometry.

Examples (from the paper directory):
  python3 revision-2026-11/experiments/pb_variants.py --suite first --variants closure --jobs 3
  python3 revision-2026-11/experiments/pb_variants.py --suite third --variants srestart --jobs 2
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import re
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "adaptive-desingularization" / "final-experiments"
SUITES = {"first": FINAL / "pb", "third": FINAL / "higher-order"}


def load_solver_module(folder: Path, name: str):
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, folder / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    return module


def restart_run(module):
    """Solver.run with S using the NC/C stopping rule (two consecutive unchanged local returns)."""
    source = textwrap.dedent(inspect.getsource(module.Solver.run))
    pattern = (r'if self\.method=="standard":\n\s+stop=termination\["classification"\]\n'
               r'\s+break\n(\s+)if self\.scan\(\):')
    source, count = re.subn(pattern, r'if self.method!="standard" and self.scan():', source)
    if count != 1:
        raise RuntimeError("restart substitution did not apply exactly once")
    namespace = {}
    exec(compile(source, f"<restart run for {module.__name__}>", "exec"), module.__dict__, namespace)
    return namespace["run"]


def closure_geometry(module):
    base = module.Geometry
    sigma = module.SIGMA
    tmin, tmax = (1e-12 * sigma) ** 2, (0.6 * sigma) ** 2

    class ClosureGeometry(base):
        """Pair half-gaps h are replaced by t = h^2 with the same feasible set of states."""
        def __init__(self, blocks, d):
            super().__init__(blocks, d)
            x, bounds, j = list(self.x), list(self.bounds), 0
            for b in self.blocks:
                j += d
                if b["kind"] == "pair":
                    j += d - 1
                    if not b["completed"]:
                        x[j] = min(max(float(x[j]) ** 2, tmin), tmax)
                        bounds[j] = (tmin, tmax)
                        j += 1
            self.x, self.bounds = np.array(x), bounds

        def unpack(self, p):
            blocks, j = [], 0
            for b, U in zip(self.blocks, self.frames):
                c = p[j:j + self.d]
                j += self.d
                new = {"id": b["id"], "kind": b["kind"], "c": c}
                if b["kind"] == "pair":
                    e = torch.as_tensor(b["e"]) + torch.as_tensor(U) @ p[j:j + self.d - 1]
                    j += self.d - 1
                    e = e / torch.linalg.vector_norm(e)
                    new.update(e=e, completed=b["completed"], h=0.0)
                    if not b["completed"]:
                        t = p[j]
                        new["h"] = torch.sqrt(t) if torch.is_tensor(t) else math.sqrt(float(t))
                        j += 1
                blocks.append(new)
            return blocks

    return ClosureGeometry


MODULES = {}


def solver_for(suite, variant):
    if suite not in MODULES:
        module = load_solver_module(SUITES[suite], f"solver_{suite}")
        MODULES[suite] = (module, module.Geometry, restart_run(module), closure_geometry(module))
    module, geometry, run_restart, geometry_closure = MODULES[suite]
    module.Geometry = geometry_closure if variant == "closure" else geometry

    class Variant(module.Solver):
        pass
    if variant == "srestart":
        Variant.run = run_restart
    return Variant


def run_task(task):
    suite, variant, args = task["suite"], task["variant"], task["args"]
    method = {"closure": "stable", "srestart": "standard"}[variant]
    Solver = solver_for(suite, variant)
    if suite == "first":
        d, g, seed, power, max_fits, qseed, reaction = args
        result = Solver(d, seed, method, power, max_fits, g, qseed, reaction).run()
    else:
        d, seed, power, max_fits, gram_order = args
        result = Solver(d, seed, method, power, max_fits, gram_order).run()
    result["variant"] = variant
    return result


def task_list(suite, variants):
    tasks = []
    for variant in variants:
        if suite == "first":
            configs = [(d, g, s) for d in (5, 10, 20) for g in (0, 1) for s in range(10, 15)]
            if variant == "srestart":   # only these S runs stopped before the work limit
                configs = [(10, 0, 12), (20, 1, 13), (5, 1, 10)]
            for d, g, s in configs:
                tasks.append({"suite": suite, "variant": variant, "name": f"d{d}_g{g}_seed{s}_{variant}",
                              "args": (d, g, s, 12, 1500, 8128, 1.0)})
        else:
            if variant != "srestart":
                raise SystemExit("third-order suite: only srestart is implemented")
            for s in range(10, 20):
                tasks.append({"suite": suite, "variant": variant, "name": f"d5_seed{s}_{variant}",
                              "args": (5, s, 12, 3000, 4)})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--variants", nargs="+", choices=["closure", "srestart"], required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or HERE / "results" / args.suite
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress").mkdir(exist_ok=True)
    os.environ["PB_PROGRESS_DIR"] = str((out / "progress").resolve())
    tasks = [t for t in task_list(args.suite, args.variants) if not (out / f"{t['name']}.json").exists()]
    sources = [SUITES[args.suite] / n for n in ("run.py", "fit.py", "kernels.py", "quadrature.py", "config.py")]
    provenance = {"command": sys.argv, "python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
                  "jobs": args.jobs, "threads_per_worker": 1, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "tasks": [t["name"] for t in tasks],
                  "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources + [Path(__file__).resolve()]}}
    (out / f"provenance-{'-'.join(args.variants)}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep going; record the failure
                (out / f"{task['name']}.error.txt").write_text(repr(exc) + "\n")
                print(f"{task['name']} FAILED {exc!r}", flush=True)
                continue
            (out / f"{task['name']}.json").write_text(json.dumps(result, indent=1) + "\n")
            print(f"{task['name']} error={result['final_relative_error']:.3e} fits={result['fits']} "
                  f"stop={result['stop']} wall={result['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
