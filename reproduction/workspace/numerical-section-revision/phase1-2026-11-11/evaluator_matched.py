#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): matched evaluator control on the first-order PB starts (d = 5, 10, 20; geometries 0, 1;
seeds 10-14).  NC with the frozen eight-node averaged evaluator, NC with the exact parity evaluator of the chart in t
(the evaluator of C), and C (= T).  Writes evaluator-matched.json."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent / "guarded-rerun-2026-11-09" / "first-order" / "accuracy"
SQ = HERE.parent / "sinum-revision-2026-11-10" / "squared-gap" / "results"
rows = []
for d in (5, 10, 20):
    for g in (0, 1):
        for s in range(10, 15):
            ex = HERE / "runs" / "evaluator" / f"pb_NC_analytic_d{d}_g{g}_s{s}.json"
            if not ex.exists():
                continue
            a = json.loads((G / f"pb_d{d}_g{g}_s{s}_stable.json").read_text())
            b = json.loads(ex.read_text())
            c = json.loads((SQ / f"pb_d{d}_g{g}_s{s}_squared_gap.json").read_text())
            rows.append(dict(d=d, g=g, seed=s, NC_avg=a["final_relative_error"], NC_exact=b["final_relative_error"],
                             C=c["final_relative_error"], fits_NC_avg=a["fits"], fits_NC_exact=b["fits"], fits_C=c["fits"]))
out = {"n": len(rows), "rows": rows}
for k in ("NC_avg", "NC_exact", "C"):
    E = [r[k] for r in rows]
    out[k] = {"median_E": st.median(E), "s6": sum(e <= 1e-6 for e in E), "s7": sum(e <= 1e-7 for e in E),
              "median_fits": st.median([r["fits_" + k] for r in rows])}
out["NC_exact_below_NC_avg"] = sum(r["NC_exact"] < r["NC_avg"] for r in rows)
out["C_below_NC_exact"] = sum(r["C"] < r["NC_exact"] for r in rows)
(HERE / "evaluator-matched.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
