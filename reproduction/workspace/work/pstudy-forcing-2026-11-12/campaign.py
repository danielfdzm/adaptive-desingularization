#!/usr/bin/env python3
"""Varying-P campaign: 2 forcings x P in {2,4,6,8,12,16} x seeds 300-309 x {S, NC, C}; plus baselines."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
import argparse, json, math, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PS = (2, 4, 6, 8, 12, 16)
SEEDS = range(300, 310)


def clean(v):
    import numpy as np
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def job(case, P, seed, method, out):
    import pstudy as ps
    path = out / "runs" / f"{case}_P{P}_s{seed}_{method}.json"
    if path.exists():
        return str(path), "exists"
    try:
        rec = ps.run_one(case, P, seed, method)
    except Exception as exc:
        rec = dict(case=case, P=P, seed=seed, label=method, stop="exception", exception=repr(exc),
                   traceback=traceback.format_exc())
    rec["sources"] = ps.source_hashes(); rec["environment"] = ps.environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(clean(rec)) + "\n"); tmp.replace(path)
    return str(path), f"{rec.get('stop')} fits={rec.get('fits')} err={rec.get('relative_H1_error')} coll={rec.get('collision')}"


def baselines(case, out):
    import pstudy as ps
    path = out / "baselines" / f"{case}.json"
    if path.exists():
        return str(path), "exists"
    rec = dict(case=case, reference=ps.reference_check(case),
               gaussian_galerkin=[ps.gaussian_galerkin(case, P) for P in PS],
               fem_p1=[ps.fem_p1(case, n) for n in (4, 8, 16, 32, 64, 128, 256, 512)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(rec), indent=1) + "\n")
    return str(path), "baselines done"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--output", type=Path, default=HERE / "results"); a = ap.parse_args()
    import pstudy as ps
    tasks = [(c, P, s, m) for P in PS for s in SEEDS for c in ps.CASES for m in ("C", "NC", "S")]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(baselines, c, a.output) for c in ps.CASES]
        futs += [ex.submit(job, c, P, s, m, a.output) for c, P, s, m in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            p, msg = f.result()
            print(f"[{i}/{len(futs)} {time.time()-t0:7.0f}s] {Path(p).name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
