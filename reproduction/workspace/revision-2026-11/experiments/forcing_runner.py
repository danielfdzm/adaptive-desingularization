#!/usr/bin/env python3
"""Run forcing-only tasks of pb-independent-1d/multi.py (loaded by path, unmodified) and store records with
non-finite floats written as strings (multi.py's strict JSON writer crashes on an inf energy change of a
rejected trial).  Usage: forcing_runner.py OUTPUT_DIR WIDTH [WIDTH ...]"""
import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"
import importlib.util, json, math, sys
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "adaptive-desingularization/final-experiments/pb-independent-1d"
sys.path.insert(0, str(D))
spec = importlib.util.spec_from_file_location("multi_forcing", D / "multi.py")
multi = importlib.util.module_from_spec(spec); spec.loader.exec_module(multi)

def clean(o):
    if isinstance(o, float) and not math.isfinite(o): return repr(o)
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for N in map(int, sys.argv[2:]):
    for seed in range(20, 25):
        for method in ("ordinary", "stable", "adaptive"):
            path = out / f"forcing_two-n{N}-{seed}-{method}.json"
            if path.exists(): continue
            try:
                r = multi.run_one("forcing_two", N, seed, method, 384, 2000)
            except Exception as exc:
                r = dict(problem="forcing_two", capacity=N, seed=seed, method=method, status="exception", exception=repr(exc))
            path.write_text(json.dumps(clean(r), indent=1))
            print(N, seed, method, r.get("error"), r.get("fits"), r.get("completed_count"), r.get("status"), flush=True)
