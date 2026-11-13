#!/usr/bin/env python3
"""Prespecified serial NC/C timing subset; run only during exclusive reservation."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import chain_experiments as ce
from restart_safe import RestartSafeSolver, VERSION
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["second1d", "higher"], required=True)
    args = parser.parse_args()
    starts = ([("second1d", 1, 0, seed) for seed in (40, 41, 42)] if args.problem == "second1d"
              else [("higher", 5, geometry, seed) for geometry, seed in ((0, 10), (0, 11), (1, 10))])
    tasks = [(*start, method) for start in starts for method in ("stable", "adaptive")]
    ce.cs.Solver = RestartSafeSolver
    derived_hashes = {"restart_safe.py": hashlib.sha256((HERE / "restart_safe.py").read_bytes()).hexdigest()}
    runner_hashes = {"run_controlled.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = HERE / f"timing-{args.problem}"
    output.mkdir(exist_ok=True)
    provenance = {"command": sys.argv, "timing_mode": "controlled", "concurrent_workers": 1,
                  "prespecified_tasks": tasks, "started_unix": time.time(),
                  "derived_solver_version": VERSION, "derived_solver_sha256": derived_hashes, "runner_source_sha256": runner_hashes,
                  "scope": "Problem/quadrature/target/solver setup plus optimization from initial fit to termination; excludes imports, process launch, mp80 endpoint validation, and JSON I/O.",
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "src").glob("*.py")},
                  "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                  "platform": platform.platform(), "machine": platform.machine(), "threadpools": threadpool_info(),
                  "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for task in tasks:
        problem, d, geometry, seed, method = task
        path = output / f"{problem}_d{d}_g{geometry}_seed{seed}_{method}.json"
        if path.exists():
            continue
        row = ce.run_one(task)
        row.update(timing_mode="controlled", concurrent_workers=1, runner_source_sha256=runner_hashes)
        accuracy_path = HERE / f"accuracy-{problem}" / path.name
        accuracy = json.loads(accuracy_path.read_text())
        fields = ("final_blocks", "final_coefficients", "events", "fits", "iterations", "joins",
                  "completions", "final_relative_error", "guard_counts", "stop")
        row["matches_frozen_accuracy"] = {key: row[key] == accuracy[key] for key in fields}
        row["matches_frozen_accuracy_all"] = all(row["matches_frozen_accuracy"].values())
        row["frozen_accuracy_sha256"] = hashlib.sha256(accuracy_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(row, indent=2) + "\n")
        print(path.name, "end_to_end_seconds", row["end_to_end_seconds"], "relative_error", row["final_relative_error"],
              "matches_frozen_accuracy", row["matches_frozen_accuracy_all"], flush=True)
        if not row["matches_frozen_accuracy_all"]:
            raise RuntimeError("Derived solver differed from frozen accuracy: " + path.name)


if __name__ == "__main__":
    main()
