#!/usr/bin/env python3
"""Frozen validation: order-two primary test and order-three robustness control.

New target shifts/amplitudes and new independent initial centers. All 60
outcomes retained. No hyperparameter changes after validation starts.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import importlib.util
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("blind1d_runner", HERE / "run1d.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_problem = runner.problem


def target_parameters(seed):
    rng = np.random.default_rng(100000 + seed)
    return float(rng.uniform(-.6, .6)), float(rng.uniform(.7, 1.3))


def validation_problem(order, seed, nodes=384, lam=10., max_fits=1200):
    p = original_problem(order, seed, nodes, lam, max_fits)
    shift, factor = target_parameters(seed)
    p["target"] = (np.array([[shift]]), np.array([[1.]]), np.array([order]),
                   np.array([factor * p["sigma"]**order]))
    return p


runner.problem = validation_problem


def run_task(task):
    result = runner.run_one(task)
    shift, factor = target_parameters(task[1])
    result.update(stage="holdout", primary_case=task[0] == 2, target_shift=shift,
                  target_amplitude_factor=factor, fixed_success_threshold=1e-6,
                  fixed_high_accuracy_threshold=1e-10)
    result["success_at_1e_6"] = result["final_relative_error"] <= 1e-6
    result["high_accuracy_at_1e_10"] = result["final_relative_error"] <= 1e-10
    return result


def main():
    out = HERE / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(order, seed, method, 384, 10., 1200)
             for order in (2, 3) for seed in range(900, 910)
             for method in ("standard_restart", "stable", "adaptive")]
    sources = [Path(__file__), HERE / "run1d.py", HERE / "run.py", runner.EXP / "chain_solver.py",
               runner.EXP / "chain_kernels.py", runner.PB / "fit.py", runner.PB / "quadrature.py"]
    manifest = dict(stage="frozen validation", primary_order=2, robustness_negative_control_order=3,
                    seeds=list(range(900, 910)), methods=["standard_restart", "stable", "adaptive"],
                    sigma=.65, reaction=10., fit_limit=1200, centre_box=3., max_order=3,
                    quadrature=dict(training_nodes=384,training_box=10.,validation_nodes=768,
                                    validation_boxes=[10.,12.]),
                    starts="iid U[-1.2,1.2], numpy.default_rng(seed), P=order+1",
                    target_generation="numpy.default_rng(100000+seed): shift U[-.6,.6], factor U[.7,1.3]; coefficient=factor*sigma**order",
                    targets=[dict(order=order, seed=seed, shift=target_parameters(seed)[0],
                                  amplitude_factor=target_parameters(seed)[1],
                                  coefficient=target_parameters(seed)[1]*.65**order,
                                  initial_centers=validation_problem(order,seed)["starts"].tolist())
                             for order in (2,3) for seed in range(900,910)],
                    success_threshold=1e-6, high_accuracy_threshold=1e-10,
                    all_results_retained=True, no_changes_after_start=True,
                    source_sha256={str(p.relative_to(runner.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    python=platform.python_version(),numpy=np.__version__,torch=runner.torch.__version__)
    pp = out / "manifest.json"
    if pp.exists() and json.loads(pp.read_text()) != runner.clean(manifest):
        raise RuntimeError("Frozen manifest differs; refusing validation run")
    runner.save(pp, manifest)
    name = lambda t: f"order{t[0]}_seed{t[1]}_{t[2]}"
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_task, t):t for t in tasks if not(out/f"{name(t)}.json").exists()}
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(stage="holdout", order=t[0], seed=t[1], method=t[2],stop="exception",exception=repr(exc))
            runner.save(out/f"{name(t)}.json",result)
            print(name(t),"error",result.get("final_relative_error"),"fits",result.get("fits"),
                  "completed",result.get("completions"),"stop",result["stop"],flush=True)


if __name__ == "__main__":
    main()
