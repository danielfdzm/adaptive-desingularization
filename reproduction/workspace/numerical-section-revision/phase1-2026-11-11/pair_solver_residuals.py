#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): stationarity residuals recorded by the first-order pair solver (no cleanup exists there).

For every first-order record in the tables (PB d = 5, 10, 20: NC, C_d frozen, C = squared-gap T; full span:
NC, C_d frozen, C Phase 1; evaluator control), report the recorded projected-gradient max-abs at the final accepted
state (clipped residual, unit scale, coefficients refitted: the same quantity as R0 of the cleanup) and the
coefficient KKT residuals.  Writes pair-solver-residuals.json.
"""
import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent / "guarded-rerun-2026-11-09" / "first-order" / "accuracy"
SQ = HERE.parent / "sinum-revision-2026-11-10" / "squared-gap" / "results"
SOURCES = [("pb", "NC", G, "pb_d*_stable.json"), ("pb", "C_d", G, "pb_d*_adaptive.json"), ("pb", "C", SQ, "pb_d*_squared_gap.json"),
           ("span", "NC", G, "linear_d*_stable.json"), ("span", "C_d", G, "linear_d*_adaptive.json"),
           ("span", "C", HERE / "runs" / "fullspan", "linear_d*_T.json"),
           ("pb_evaluator_control", "NC_analytic", HERE / "runs" / "evaluator", "pb_NC_analytic_*.json"),
           ("pb_evaluator_control", "C_d_analytic", HERE / "runs" / "evaluator", "pb_Cd_analytic_*.json"),
           ("span", "NC_exact", HERE / "runs" / "evaluator-linear", "linear_stable_exact_*.json"),
           ("span", "C_d_exact", HERE / "runs" / "evaluator-linear", "linear_adaptive_exact_*.json")]
rows = []
for cohort, method, base, pat in SOURCES:
    for f in sorted(glob.glob(str(base / pat))):
        r = json.loads(Path(f).read_text())
        cf = r.get("coefficient_fit") or {}
        rows.append({"file": str(Path(f).relative_to(HERE.parent)), "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(),
                     "cohort": cohort, "method": method, "d": r.get("dimension", r.get("d")), "geometry": r.get("geometry"),
                     "seed": r.get("seed"), "stop": r.get("stop"), "final_relative_error": r.get("final_relative_error"),
                     "projected_gradient_max_abs": r.get("projected_gradient_max_abs"),
                     "kkt_residual": cf.get("kkt_residual"), "kkt_scaled_residual": cf.get("kkt_scaled_residual")})
(HERE / "pair-solver-residuals.json").write_text(json.dumps(rows, indent=1, default=float))
from collections import defaultdict
import statistics as st
g = defaultdict(list)
for r in rows:
    g[(r["cohort"], r["method"])].append(r)
for k, v in sorted(g.items()):
    pg = [x["projected_gradient_max_abs"] for x in v if x["projected_gradient_max_abs"] is not None]
    kk = [x["kkt_scaled_residual"] for x in v if x["kkt_scaled_residual"] is not None]
    print(k, len(v), "pg median %.1e max %.1e" % (st.median(pg), max(pg)) if pg else "", "kkt max %.1e" % max(kk) if kk else "")
