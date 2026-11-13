#!/usr/bin/env python3
"""Rebuild current main-paper Table 1 directly from its matched NC/C records.

Uses all 100 starts per derivative order and all 50 starts per PB dimension.
No optimization or numerical-library dependency is required.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
INPUTS = {}


def read(relative):
    path = BASE / relative
    raw = path.read_bytes()
    INPUTS[relative] = hashlib.sha256(raw).hexdigest()
    row = json.loads(raw)
    error, fits = row["final_relative_error"], row["fits"]
    if not math.isfinite(error) or error < 0 or fits < 0:
        raise ValueError(f"Invalid endpoint: {relative}")
    return {"error": error, "fits": fits}


def summarize(label, methods, expected):
    if any(len(rows) != expected for rows in methods.values()):
        raise ValueError(f"Incomplete {label}: expected {expected} matched starts")
    result = {"study": label, "n": expected}
    for method, rows in methods.items():
        result[f"error_{method}"] = median(row["error"] for row in rows)
        result[f"successes_{method}"] = sum(row["error"] <= 1e-6 for row in rows)
        result[f"fits_{method}"] = median(row["fits"] for row in rows)
    return result


def scientific(value, digits, latex=False):
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    if latex:
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{mantissa}e{int(exponent):+d}"


def main():
    results = []
    for order in (2, 3):
        methods = {method: [read(f"cohort1d/results_amend2/order{order}_seed{seed}_{method}.json")
                            for seed in range(900, 1000)] for method in ("NC", "C")}
        results.append(summarize(f"Independent, r={order}", methods, 100))
    for dimension in (5, 10, 20):
        methods = {method: [read(f"cohort_pb/{directory}/pb_d{dimension}_g{geometry}_s{seed}_{suffix}.json")
                            for geometry in (0, 1) for seed in range(10, 35)]
                   for method, directory, suffix in (("NC", "results_nc_exact", "NCx"),
                                                     ("C", "results", "T"))}
        results.append(summarize(f"PB, d={dimension}", methods, 50))

    with (HERE / "table-main-recovery.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    markdown = ["# Main paper, Table 1", "",
                "All starts, including failures; success means relative H1 error ≤ 1e-6.", "",
                "| Study | n | Error NC | Error C | Successes NC/C | Fits NC/C |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
    latex = [r"\begin{tabular}{l r c c r r}", r"\toprule",
             r"Study & $n$ & $E_{\rm NC}$ & $E_{\rm C}$ & Successes NC/C & Fits NC/C\\",
             r"\midrule"]
    for row in results:
        digits = 3 if row["study"].startswith("Independent") else 2
        common = [str(row["n"]), scientific(row["error_NC"], digits), scientific(row["error_C"], digits),
                  f"{row['successes_NC']}/{row['successes_C']}", f"{row['fits_NC']:g}/{row['fits_C']:g}"]
        markdown.append("| " + " | ".join([row["study"], *common]) + " |")
        label = row["study"].replace("r=", "$r=").replace("d=", "$d=") + "$"
        cells = [label, common[0], scientific(row["error_NC"], digits, True),
                 scientific(row["error_C"], digits, True), *common[3:]]
        latex.append(" & ".join(cells) + r"\\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / "table-main-recovery.md").write_text("\n".join(markdown) + "\n")
    (HERE / "table-main-recovery.tex").write_text("\n".join(latex) + "\n")
    (HERE / "table-main-recovery.json").write_text(json.dumps({
        "manuscript_label": "tab:main-recovery", "success_threshold": 1e-6,
        "rows": results, "input_sha256": INPUTS,
    }, indent=2) + "\n")
    print("\n".join(markdown))
    print(f"\nRead {len(INPUTS)} matched raw endpoints; wrote CSV, Markdown, LaTeX and provenance JSON.")


if __name__ == "__main__":
    main()
