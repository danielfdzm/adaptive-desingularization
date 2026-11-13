#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): summaries for the revised tables (S, NC, C = revised Algorithm 1, C_d = ablation).

Sources (all read-only):
  runs/                                   Phase 1 records (C on chain cohorts, full span, variance, 2D; 1D linear S/NC/C_d/C)
  ../guarded-rerun-2026-11-09/...         frozen NC/C_d records and historical S baselines
  ../sinum-revision-2026-11-10/squared-gap/results   T records = C for first-order PB (pairs only)
  ../../work/sinum-phase3-2026-11-11/s_reruns/results.json   S with the two-return rule, second order (Phase 3)
Writes summary.json and table-rows.tex.  Success means relative H1 error <= 1e-6.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent / "guarded-rerun-2026-11-09"
SQ = HERE.parent / "sinum-revision-2026-11-10" / "squared-gap" / "results"
P3 = HERE.parents[1] / "work" / "sinum-phase3-2026-11-11" / "s_reruns" / "results.json"


def load(path):
    return json.loads(Path(path).read_text())


def stats(rows, key="final_relative_error", fits="fits"):
    E = [r[key] for r in rows]
    F = [r[fits] for r in rows]
    return {"n": len(E), "median_error": st.median(E), "success_1e6": sum(e <= 1e-6 for e in E),
            "success_1e10": sum(e <= 1e-10 for e in E), "median_fits": st.median(F),
            "errors": E, "fits": F}


def chain(pattern):
    return [load(f) for f in sorted(glob.glob(str(pattern)))]


def phase3_s(problem):
    data = load(P3)
    rows = data[problem] if isinstance(data, dict) and problem in data else []
    out = []
    for r in rows:
        if isinstance(r, dict) and "final_relative_error" in r:
            out.append(r)
    return out


def main():
    S = {}
    # 1D linear (directional solver, lambda = 0): all four methods are Phase 1 runs on the same starts.
    S["linear1d"] = {m: stats(chain(HERE / f"runs/linear1d/linear1d_*_{k}.json"))
                     for m, k in (("S", "standard_restart"), ("NC", "stable"), ("C_d", "adaptive"), ("C", "tpair"))}
    # second order, d = 1: S (two-return rule, Phase 3), NC/C_d frozen, C Phase 1
    s_second = phase3_s("second1d")
    S["second1d"] = {"S": stats(s_second) if s_second else None,
                     "NC": stats(chain(G / "directional/accuracy-second1d/second1d_*_stable.json")),
                     "C_d": stats(chain(G / "directional/accuracy-second1d/second1d_*_adaptive.json")),
                     "C": stats(chain(HERE / "runs/second1d/second1d_*_tpair.json"))}
    # higher order, d = 5
    S["higher_d5"] = {"S": stats(chain(G / "historical-baselines/higher_d5_*_standard.json")),
                      "NC": stats(chain(G / "directional/accuracy-higher/higher_*_stable.json")),
                      "C_d": stats(chain(G / "directional/accuracy-higher/higher_*_adaptive.json")),
                      "C": stats(chain(HERE / "runs/higher/higher_*_tpair.json"))}
    # first-order PB, d = 5, 10, 20 (pairs only: C = T)
    for d in (5, 10, 20):
        S[f"pb_d{d}"] = {"S": stats(chain(G / f"historical-baselines/pb_d{d}_*_standard.json")),
                         "NC": stats(chain(G / f"first-order/accuracy/pb_d{d}_*_stable.json")),
                         "C_d": stats(chain(G / f"first-order/accuracy/pb_d{d}_*_adaptive.json")),
                         "C": stats(chain(SQ / f"pb_d{d}_*_squared_gap.json"))}
    # full-span linear, d = 10, 20
    for d in (10, 20):
        S[f"span_d{d}"] = {"S": stats(chain(G / f"historical-baselines/linear_d{d}_*_standard.json")),
                           "NC": stats(chain(G / f"first-order/accuracy/linear_d{d}_*_stable.json")),
                           "C_d": stats(chain(G / f"first-order/accuracy/linear_d{d}_*_adaptive.json")),
                           "C": stats(chain(HERE / f"runs/fullspan/linear_d{d}_*_T.json"))}
    # variance family (Table 3)
    ref = {0.99: 7.68848e-3, 0.995: 3.83423e-3, 0.999: 7.65251e-4}
    var = {}
    for s in (0.99, 0.995, 0.999):
        var[str(s)] = {}
        for m, pat in (("S", G / f"variance/results/s{s:g}_seed*_S.json"),
                       ("NC", G / f"variance/results/s{s:g}_seed*_NC.json"),
                       ("C_d", G / f"variance/results/s{s:g}_seed*_C.json"),
                       ("C", HERE / f"runs/variance/s{s:g}_seed*_C.json")):
            rows = chain(pat)
            E = [r["final"]["relative_error"] for r in rows]
            var[str(s)][m] = {"n": len(rows), "within_1pct": sum(abs(e - ref[s]) <= 0.01 * ref[s] for e in E),
                              "median_error": st.median(E), "median_iterations": st.median([r["iterations"] for r in rows]),
                              "median_seconds": st.median([r["optimizer_wall_seconds"] for r in rows]),
                              "median_stored_norm": st.median([r["final"]["stored_coefficient_norm"] for r in rows]),
                              "zero_gap": sum(bool(r.get("zero_gap", r["final"].get("h", 1) == 0.)) for r in rows)}
    S["variance"] = var
    # two-dimensional controls
    twod = {}
    for cohort in ("recovery", "mismatch001", "mismatch005", "pilot-smoke", "pilot-smooth", "pilot-multi6"):
        rows = {}
        for m, key in (("NC", "stable"), ("C_d", "adaptive")):
            rows[m] = [load(f) for f in sorted(glob.glob(str(G / f"supplementary-2d/accuracy/{cohort}_*_{key}.json")))]
        rows["C"] = [load(f) for f in sorted(glob.glob(str(HERE / f"runs/twod/{cohort}_*_stable_T.json")))]
        def err(r):
            ep = r.get("independent_endpoint_check", {})
            return ep.get("relative_H1", r.get("final_relative_error"))
        twod[cohort] = {m: {"n": len(v), "median_error": st.median([err(r) for r in v]) if v else None,
                            "median_fits": st.median([r["fits"] for r in v]) if v else None,
                            "success_1e6": sum(err(r) <= 1e-6 for r in v)} for m, v in rows.items()}
    S["twod"] = twod
    # evaluator control (first-order PB, seeds 10-14)
    ev = {}
    for m in ("NC_analytic", "Cd_analytic"):
        rows = chain(HERE / f"runs/evaluator/pb_{m}_*.json")
        if rows:
            ev[m] = stats(rows)
    S["evaluator_control"] = ev
    (HERE / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    # compact print
    for k, v in S.items():
        if k in ("variance", "twod", "evaluator_control"):
            continue
        line = f"{k:10s}"
        for m in ("S", "NC", "C", "C_d"):
            x = v.get(m)
            if x:
                line += f" | {m}: {x['median_error']:.1e} {x['success_1e6']}/{x['n']} f={x['median_fits']}"
        print(line)
    for s, v in S["variance"].items():
        print("variance", s, {m: (x["within_1pct"], f"{x['median_error']:.5e}", x["median_iterations"]) for m, x in v.items()})
    for c, v in S["twod"].items():
        print("2D", c, {m: (x["n"], x["median_error"] and f"{x['median_error']:.3e}", x["median_fits"]) for m, x in v.items()})
    for m, x in ev.items():
        print("evaluator", m, f"{x['median_error']:.2e} {x['success_1e6']}/{x['n']} f={x['median_fits']}")


if __name__ == "__main__":
    main()
