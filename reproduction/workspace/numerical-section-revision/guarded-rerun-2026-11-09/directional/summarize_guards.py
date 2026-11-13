"""Per-case guard chronology and final distance to the chart cap."""
from pathlib import Path
import collections
import json
import numpy as np

HERE = Path(__file__).resolve().parent
rows = []
for path in sorted((HERE / "accuracy-higher").glob("higher_*.json")):
    run = json.loads(path.read_text())
    completions = [event for event in run["events"] if event["kind"] == "completion"]
    first_completion = min((event["fits"] for event in completions), default=None)
    examples = run.get("guard_examples", [])
    first_guard = min((event["fits"] for event in examples), default=None)
    if first_guard is None:
        relation = "no_guard_rejection"
    elif first_completion is None:
        relation = "guard_rejection_with_no_completion"
    elif first_guard < first_completion:
        relation = "before_first_completion"
    elif first_guard == first_completion:
        relation = "same_fit_counter_order_not_resolved"
    else:
        relation = "after_first_completion"
    ratios = []
    for block in run["final_blocks"]:
        if block["kind"] != "line":
            continue
        base = [block["h"], -block["h"]] if block["base"] == "pair" else [0.] * (block["kb"] + 1)
        rho = np.ptp(np.asarray(base + block["s"]) / .45)
        cap = min(1., float(rho))
        ratios.extend((np.linalg.norm(block["p"], axis=1) / cap).tolist())
    rows.append({"file": path.name, "geometry": run["geometry"], "seed": run["seed"], "method": run["method"],
                 "relative_error": run["final_relative_error"], "fits": run["fits"], "stop": run["stop"],
                 "completions": run["completions"], "guard_counts": run.get("guard_counts", {}),
                 "first_recorded_guard_fit": first_guard, "first_completion_fit": first_completion,
                 "guard_chronology": relation,
                 "all_guard_events_recorded": sum(run.get("guard_counts", {}).values()) <= len(examples),
                 "final_max_p_hat_over_cap": max(ratios, default=None)})
report = {"complete": len(rows) == 20, "expected_rows": 20, "runs": len(rows),
          "chronology_counts": dict(collections.Counter(row["guard_chronology"] for row in rows)),
          "interpretation": "A local return near the chart cap is optimization behavior. These errors do not characterize the best attainable approximation.",
          "rows": rows}
(HERE / "guard-chronology.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
