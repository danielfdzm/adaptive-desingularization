#!/usr/bin/env python3
"""All 150 starts of the first-order PB cohort x four arms; keeps existing outputs."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
import argparse, json, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["pg_t", "lbfgs_h", "lbfgs_t", "pg_h"]


def job(d, g, s, arm, tol, max_fits, alpha0, out):
    import alg41
    path = out / arm / f"d{d}_g{g}_s{s}.json"
    if path.exists():
        return str(path), "exists"
    try:
        rec = alg41.run_one(d, g, s, arm, tol, max_fits, alpha0)
    except Exception as exc:          # every outcome is retained
        rec = dict(d=d, geometry=g, seed=s, arm=arm, stop="exception", exception=repr(exc),
                   traceback=traceback.format_exc())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, allow_nan=True) + "\n")
    tmp.replace(path)
    return str(path), f"{rec.get('stop')} fits={rec.get('fits')} res={rec.get('final_residual')} E={rec.get('final_relative_error_110')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-fits", type=int, default=2500)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=HERE / "results")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    # Rescoped after the first 17 runs (cost): pg_t, lbfgs_t, lbfgs_h on all 150 starts;
    # pg_h (comparator for the chart only) on seeds 10-14, i.e. 30 starts.
    # Final design (amendments 1-2, protocol.json): pg_t on seeds 10-14, pg_h on seeds 10-12,
    # lbfgs_t and lbfgs_h on all 150 starts.
    limit = {"pg_t": 14, "pg_h": 12, "lbfgs_t": 34, "lbfgs_h": 34}
    tasks = [(d, g, s, arm) for s in range(10, 35) for d in (5, 10, 20) for g in (0, 1)
             for arm in ARMS if s <= limit[arm]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(job, d, g, s, arm, a.tol, a.max_fits, a.alpha0, a.output) for d, g, s, arm in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            p, msg = f.result()
            print(f"[{i}/{len(tasks)} {time.time()-t0:7.0f}s] {Path(p).parent.name}/{Path(p).name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
