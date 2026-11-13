"""Summarize matched guarded cohorts without treating concurrent wall times as benchmarks."""
from pathlib import Path
import collections
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
report = {}
for name, old_folder in (("second1d", "chain-second1d-final"), ("higher", "chain-higher-d5")):
    runs = [json.loads(p.read_text()) for p in (HERE / f"accuracy-{name}").glob(f"{name}_*.json")]
    historical = ROOT / "revision-2026-11/experiments/results" / old_folder
    runs += [json.loads(p.read_text()) for p in historical.glob(f"{name}_*_standard.json")]
    group = {}
    for method in ("standard", "stable", "adaptive"):
        rows = [r for r in runs if r["method"] == method]
        if not rows:
            continue
        errors = [r["final_relative_error"] for r in rows]
        guards = collections.Counter()
        for r in rows:
            guards.update(r.get("guard_counts", {}))
        group[method] = {"n": len(rows), "historical_unchanged_baseline": method == "standard",
                         "median_relative_error": float(np.median(errors)), "min_relative_error": min(errors),
                         "max_relative_error": max(errors), "success_below_1e_minus_6": sum(x < 1e-6 for x in errors),
                         "median_fits": float(np.median([r["fits"] for r in rows])),
                         "median_joins": float(np.median([r["joins"] for r in rows])),
                         "median_completions": float(np.median([r["completions"] for r in rows])),
                         "guard_rejections": dict(guards), "termination_reasons": dict(collections.Counter(r["stop"] for r in rows))}
    by_start = {}
    for r in runs:
        by_start.setdefault((r.get("geometry", 0), r["seed"]), {})[r["method"]] = r
    paired = [v for v in by_start.values() if "stable" in v and "adaptive" in v]
    group["paired"] = {"n": len(paired), "C_error_below_NC": sum(v["adaptive"]["final_relative_error"] < v["stable"]["final_relative_error"] for v in paired),
                       "rows": [{"geometry": key[0], "seed": key[1], **{m: r["final_relative_error"] for m, r in v.items()}} for key, v in sorted(by_start.items())]}
    report[name] = group
(HERE / "accuracy-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
