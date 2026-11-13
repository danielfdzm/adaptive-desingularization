#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): every number that the revised Section 5 quotes, computed from the records.

Writes paper-numbers.json (and prints it).  Sources as in summarize_phase1.py, plus the Phase 3 independent
cohort records for seeds 900-909 (work/sinum-phase3-2026-11-11/cohort1d/results; their S/NC/C records
reproduce the frozen guarded records, "C" there is C_d here and "tpair" is C).
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
G = HERE.parent / "guarded-rerun-2026-11-09"
SQ = HERE.parent / "sinum-revision-2026-11-10" / "squared-gap" / "results"
P3 = ROOT / "work" / "sinum-phase3-2026-11-11"


def load(p):
    return json.loads(Path(p).read_text())


def recs(pattern):
    return [load(f) for f in sorted(glob.glob(str(pattern)))]


def err(r):
    if "final" in r and isinstance(r["final"], dict):
        return r["final"]["relative_error"]
    ep = r.get("independent_endpoint_check")
    if isinstance(ep, dict) and "relative_H1" in ep:
        return ep["relative_H1"]
    if "final_relative_error" not in r and "error" in r:      # Phase 3 S reruns (s_reruns/results.json)
        return r["error"]
    return r["final_relative_error"]


def summary(rows, fits="fits"):
    if not rows:
        return None
    E = [err(r) for r in rows]
    out = {"n": len(E), "median_E": st.median(E), "s6": sum(e <= 1e-6 for e in E), "s10": sum(e <= 1e-10 for e in E),
           "max_E": max(E), "min_E": min(E)}
    if fits and all(fits in r for r in rows):
        out["median_fits"] = st.median([r[fits] for r in rows])
    return out


AMEND_SOURCES = []


def main():
    N = {}
    # Table 1A: independent cohort, seeds 900-909
    for order in (2, 3):
        rows = {}
        # results_amend2/ uses the paper labels (C = tpair, Cd = C_d); results/ the solver suffixes.
        for m, new_key, old_key in (("S", "S", "S"), ("NC", "NC", "NC"), ("C", "C", "tpair"), ("C_d", "Cd", "C")):
            recs1 = []
            for s in range(900, 910):
                amended = P3 / f"cohort1d/results_amend2/order{order}_seed{s}_{new_key}.json"   # amendment 2
                recs1.append(load(amended if amended.exists() else P3 / f"cohort1d/results/order{order}_seed{s}_{old_key}.json"))
                AMEND_SOURCES.append(str(amended.exists()))
            rows[m] = summary(recs1)
        N[f"table1A_r{order}"] = rows
    # Table 2 rows
    T2 = {}
    T2["linear1d"] = {m: summary(recs(HERE / f"runs/linear1d/*_{k}.json"))
                      for m, k in (("S", "standard_restart"), ("NC", "stable"), ("C", "tpair"), ("C_d", "adaptive"))}
    for d in (5, 10, 20):
        T2[f"pb_d{d}"] = {"S": summary(recs(G / f"historical-baselines/pb_d{d}_*_standard.json")),
                          "NC": summary(recs(G / f"first-order/accuracy/pb_d{d}_*_stable.json")),
                          "C": summary(recs(SQ / f"pb_d{d}_*_squared_gap.json")),
                          "C_d": summary(recs(G / f"first-order/accuracy/pb_d{d}_*_adaptive.json"))}
    s_second = [r for r in load(P3 / "s_reruns/results.json").get("second1d", []) if isinstance(r, dict)]
    T2["second1d"] = {"S": summary(s_second),
                      "NC": summary(recs(G / "directional/accuracy-second1d/*_stable.json")),
                      "C": summary(recs(HERE / "runs/second1d/*_tpair.json")),
                      "C_d": summary(recs(G / "directional/accuracy-second1d/*_adaptive.json"))}
    T2["higher_d5"] = {"S": summary(recs(G / "historical-baselines/higher_d5_*_standard.json")),
                       "NC": summary(recs(G / "directional/accuracy-higher/*_stable.json")),
                       "C": summary(recs(HERE / "runs/higher/*_tpair.json")),
                       "C_d": summary(recs(G / "directional/accuracy-higher/*_adaptive.json"))}
    for d in (10, 20):
        T2[f"span_d{d}"] = {"S": summary(recs(G / f"historical-baselines/linear_d{d}_*_standard.json")),
                            "NC": summary(recs(G / f"first-order/accuracy/linear_d{d}_*_stable.json")),
                            "C": summary(recs(HERE / f"runs/fullspan/linear_d{d}_*_T.json")),
                            "C_d": summary(recs(G / f"first-order/accuracy/linear_d{d}_*_adaptive.json"))}
    N["table1A_amended_records"] = AMEND_SOURCES.count("True")
    # basin differences between C and C_d on seeds 900-909 (amended records, threshold 1e-6)
    diff = {}
    for order in (2, 3):
        e = {}
        for m, key in (("C", "C"), ("C_d", "Cd")):
            e[m] = {s: err(load(P3 / f"cohort1d/results_amend2/order{order}_seed{s}_{key}.json")) for s in range(900, 910)}
        diff[str(order)] = {"C_only_fail": [s for s in e["C"] if e["C"][s] > 1e-6 >= e["C_d"][s]],
                            "Cd_only_fail": [s for s in e["C"] if e["C_d"][s] > 1e-6 >= e["C"][s]]}
    N["table1A_basin_diff"] = diff
    N["table2"] = T2
    # first-order pooled facts for C (= T) versus C_d
    T = recs(SQ / "pb_d*_squared_gap.json")
    N["first_order_C"] = {"zero_pairs": sum(r.get("zero_squared_gaps", 0) for r in T),
                          "pairs": sum(len(r.get("squared_gaps", [])) for r in T),
                          "runs_with_zero": sum(r.get("zero_squared_gaps", 0) > 0 for r in T),
                          "median_fits": st.median([r["fits"] for r in T]),
                          "median_fits_Cd": st.median([r["fits"] for r in recs(G / "first-order/accuracy/pb_d*_adaptive.json")]),
                          "fits_range": [min(r["fits"] for r in T), max(r["fits"] for r in T)]}
    # higher order and second order details for C
    for name, pat in (("higher", HERE / "runs/higher/*_tpair.json"), ("second", HERE / "runs/second1d/*_tpair.json"),
                      ("linear1d", HERE / "runs/linear1d/*_tpair.json")):
        rows = recs(pat)
        N[f"{name}_C_events"] = {"completions": sum(r["completions"] for r in rows), "rejections": sum(r["rejections"] for r in rows),
                                 "slopes": [e.get("slope_normalized") for r in rows for e in r["events"] if e["kind"] in ("completion", "rejected") and "slope_normalized" in e],
                                 "slope_rejections": sum(1 for r in rows for e in r["events"] if e.get("slope_ok") is False),
                                 "final_chain_slopes_min": min([s["slope_normalized"] for r in rows for s in r["stationarity"].get("final_chain_slopes", [])] or [float("nan")]),
                                 "zero_gap_pairs_end": sum(r.get("zero_gap_pairs", 0) for r in rows),
                                 "pairs_end": sum(len(r.get("final_squared_halfgaps", [])) for r in rows),
                                 "below_1e10": sum(err(r) <= 1e-10 for r in rows), "n": len(rows)}
    # variance family
    ref = {0.99: 7.68848e-3, 0.995: 3.83423e-3, 0.999: 7.65251e-4}
    V = {}
    for s in (0.99, 0.995, 0.999):
        V[str(s)] = {}
        for m, pat in (("S", G / f"variance/results/s{s:g}_seed*_S.json"), ("NC", G / f"variance/results/s{s:g}_seed*_NC.json"),
                       ("C", HERE / f"runs/variance/s{s:g}_seed*_C.json"), ("C_d", G / f"variance/results/s{s:g}_seed*_C.json")):
            rows = recs(pat)
            E = [r["final"]["relative_error"] for r in rows]
            V[str(s)][m] = {"n": len(rows), "runs": sum(abs(e - ref[s]) <= .01 * ref[s] for e in E), "median_E": st.median(E),
                            "iter": st.median([r["iterations"] for r in rows]), "time": st.median([r["optimizer_wall_seconds"] for r in rows]),
                            "norm": st.median([r["final"]["stored_coefficient_norm"] for r in rows]),
                            "failed_seeds": sorted(r["seed"] for r, e in zip(rows, E) if abs(e - ref[s]) > .01 * ref[s]),
                            "zero_gap": sum(1 for r in rows if r.get("zero_gap")) if m == "C" else None}
    N["variance"] = V
    # two-dimensional controls
    TW = {}
    for cohort in ("recovery", "mismatch001", "mismatch005", "pilot-smoke", "pilot-smooth", "pilot-multi6"):
        TW[cohort] = {"NC": summary(recs(G / f"supplementary-2d/accuracy/{cohort}_*_stable.json")),
                      "C_d": summary(recs(G / f"supplementary-2d/accuracy/{cohort}_*_adaptive.json")),
                      "C": summary(recs(HERE / f"runs/twod/{cohort}_*_stable_T.json"))}
    N["twod"] = TW
    # evaluator control
    N["evaluator"] = {m: summary(recs(HERE / f"runs/evaluator/pb_{m}_*.json")) for m in ("NC_analytic", "Cd_analytic")}
    (HERE / "paper-numbers.json").write_text(json.dumps(N, indent=1, default=float))
    print(json.dumps(N, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
