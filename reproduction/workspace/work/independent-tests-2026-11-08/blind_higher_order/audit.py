#!/usr/bin/env python3
"""Check source provenance, pairing, coverage, and endpoint validation."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
reports=[]
for campaign,expected in (("pilot",18),("pilot_strong",18),("pilot1d",36),("holdout",60)):
    directory=HERE/campaign
    manifest=directory/("manifest.json" if campaign=="holdout" else "protocol.json")
    if not manifest.exists():continue
    protocol=json.loads(manifest.read_text())
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol["source_sha256"].items()}
    records=[json.loads(p.read_text()) for p in sorted(directory.glob("order*_seed*.json"))]
    paired=[]
    for order,seed in sorted({(r["order"],r["seed"]) for r in records}):
        g=[r for r in records if r["order"]==order and r["seed"]==seed]
        paired.append(dict(order=order,seed=seed,methods=len(g),
                           same_starts=len(g)==3 and all(r.get("initial_centers")==g[0].get("initial_centers") for r in g),
                           same_target=len(g)==3 and all(r.get("target")==g[0].get("target") for r in g)))
    validation=[]
    for r in records:
        if "validation" not in r:continue
        qs=r["validation"]["quadrature"]
        e=r["final_relative_error"]
        target2=2*qs[-1]["quadratic_gap"]/e**2 if e else 1.
        absolute=max(abs(q["total_gap"]-qs[-1]["total_gap"]) for q in qs[:-1])
        validation.append(dict(order=r["order"],seed=r["seed"],method=r["method"],error=e,
                               max_absolute_gap_change=absolute,max_gap_change_over_target_norm2=absolute/target2,
                               relative_gap_change=r["validation"]["maximum_relative_gap_change"],
                               nonlinear_over_quadratic=qs[-1]["target_nonlinear_over_quadratic"],
                               coefficient_kkt=r["validation"]["normalized_coefficient_kkt"]))
    reports.append(dict(campaign=campaign,expected=expected,found=len(records),complete=len(records)==expected,
                        source_hashes_match=hashes,pair_checks=paired,validation=validation,
                        max_gap_change_over_target_norm2=max((v["max_gap_change_over_target_norm2"] for v in validation),default=None),
                        nonlinear_fraction_range=[min(v["nonlinear_over_quadratic"] for v in validation),
                                                  max(v["nonlinear_over_quadratic"] for v in validation)] if validation else None))
result=dict(campaigns=reports,all_complete=all(r["complete"] for r in reports),
            all_source_hashes_match=all(all(r["source_hashes_match"].values()) for r in reports),
            all_pairings_match=all(all(p["same_starts"] and p["same_target"] for p in r["pair_checks"]) for r in reports),
            caveat="Relative energy validation error is unstable at gaps near roundoff squared; use absolute gap change and 80-digit H1 error.",
            excluded_from_numerical_campaigns="smoke/ (20-fit implementation check) and pilot1d-launchfailure/ (worker import failed before optimization)")
(HERE/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="campaigns"},indent=2))
for r in reports:
    print(r["campaign"],r["found"],r["max_gap_change_over_target_norm2"],r["nonlinear_fraction_range"])
