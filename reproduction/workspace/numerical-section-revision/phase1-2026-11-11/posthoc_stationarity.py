#!/usr/bin/env python3
"""Phase 1/2 (11 Nov 2026): post-hoc stationarity of recorded chain-solver endpoints.

For every record of the given run directories (chain_experiments.py output), rebuild the solver at the
recorded final state (final_blocks, final_coefficients; problem data from chain_experiments), then call the
Phase 2 frozen-chart cleanup (src/safeguard_cleanup.py).  Its entry residual "initial.pg_max" is the
projected-gradient residual R = x - clip(x - g, lo, hi) (gamma = 1, unit scale) of the paper's stationarity
theorem; its exit says whether the endpoint is stationary to tol in the frozen chart or is a guard,
rounding or work-limit stop.  The recorded main endpoints are not changed.
Usage: python3.13 posthoc_stationarity.py runs/linear1d runs/second1d runs/higher --output posthoc-stationarity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import chain_solver as cs                     # noqa: E402
import chain_kernels as ck                    # noqa: E402
import chain_experiments as ce                # noqa: E402
import safeguard_cleanup as sc                # noqa: E402


def problem_for(rec):
    if rec["problem"] == "linear1d":
        return ce.problem_linear1d(rec["seed"])
    if rec["problem"] == "second1d":
        return ce.problem_second1d(rec["seed"])
    return ce.problem_higher(rec["d"], rec["geometry"], rec["seed"])


def rebuild(rec):
    P = problem_for(rec)
    s = cs.Solver(P["d"], P["sigma"], P["target"], P["starts"], rec["method"], P["X"], P["W"],
                  lam=P.get("lam", 1.0), max_fits=10 ** 9, box=P["box"])
    blocks = []
    for b in rec["final_blocks"]:
        nb = {}
        for k, v in b.items():
            nb[k] = np.asarray(v, dtype=float) if k in ("c", "e", "s", "p") else v
        if "p" in nb:
            nb["p"] = nb["p"].reshape(-1, P["d"])
        blocks.append(nb)
    s.blocks = blocks
    s.coef = np.asarray(rec["final_coefficients"], dtype=float)
    s.warm = s.coef.copy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("provenance"):
                continue
            rec = json.loads(f.read_text())
            if "final_blocks" not in rec:                     # not a run record
                continue
            s = rebuild(rec)
            s.fits = 0
            s.max_fits = args.max_evals
            out = sc.cleanup(s, tol=args.tol, max_evals=args.max_evals)
            after = None
            if out.get("iterations", 0) > 0:                   # 80-digit endpoint error after cleanup steps
                A = s.state_primitives()
                B = [(c, u, o, -w) for c, u, o, w in s.target_np]
                err2 = ck.mp_norm2(A + B, s.sigma, s.d, dps=80)
                after = float((max(err2, 0) / ck.mp_norm2(s.target_np, s.sigma, s.d, dps=80)) ** 0.5)
            init, fin = out.get("initial", {}), out.get("final", {})
            rows.append({"file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                         "cohort": rec["problem"], "method": rec["method"], "d": rec.get("d"),
                         "geometry": rec.get("geometry"), "seed": rec["seed"], "stop_main": rec["stop"],
                         "R0": init.get("pg_max"), "objective_noise": out.get("objective_noise"),
                         "cleanup_stop": out.get("stop"), "R_final": fin.get("pg_max"),
                         "cleanup_iterations": out.get("iterations"), "cleanup_evaluations": out.get("evaluations"),
                         "error_before_80_digits": rec["final_relative_error"], "error_after_80_digits": after,
                         "recorded_pg_at_bounds": rec.get("stationarity", {}).get("projected_gradient_max_abs"),
                         "cleanup": out})
            print(f"{f.name}: R0={init.get('pg_max')} -> {fin.get('pg_max')} stop={out.get('stop')} "
                  f"dE={fin.get('rel_error_double', float('nan'))}", flush=True)
    args.output.write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
