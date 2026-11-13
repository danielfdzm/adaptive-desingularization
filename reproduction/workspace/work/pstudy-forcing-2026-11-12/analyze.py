#!/usr/bin/env python3
"""Summaries, LaTeX table rows and the figure of the varying-P forcing study (records only)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "adaptive-desingularization" / "experiments"))
import plot_style as ps_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CASES = ("single_source", "sech_dipole")
CASE_TITLES = {"single_source": r"(a) Odd single-source forcing, $\lambda=10$",
               "sech_dipole": r"(b) Asymmetric sech$^2$ forcing, $\lambda=5$"}
PS = (2, 4, 6, 8, 12, 16)
METHODS = ("S", "NC", "C")


def load():
    runs = [json.loads(p.read_text()) for p in sorted((HERE / "results" / "runs").glob("*.json"))]
    base = {c: json.loads((HERE / "results" / "baselines" / f"{c}.json").read_text()) for c in CASES}
    return runs, base


def fem_dofs_to_match(fem, err):
    """Smallest P1 dof count (log-log interpolation) with error <= err."""
    n = np.array([f["dof"] for f in fem], float)
    e = np.array([f["relative_H1_error"] for f in fem], float)
    if err >= e[0]:
        return float(n[0])
    if err < e[-1]:
        return None
    for i in range(len(n) - 1):
        if e[i] >= err >= e[i + 1]:
            s = (math.log(err) - math.log(e[i])) / (math.log(e[i + 1]) - math.log(e[i]))
            return float(math.exp(math.log(n[i]) + s * (math.log(n[i + 1]) - math.log(n[i]))))
    return None


def summarize(runs, base):
    out = {}
    for c in CASES:
        out[c] = {}
        for P in PS:
            row = {}
            for m in METHODS:
                rr = [r for r in runs if r["case"] == c and r["P"] == P and r.get("label") == m]
                ok = [r for r in rr if isinstance(r.get("relative_H1_error"), (int, float))
                      and r["relative_H1_error"] <= 1.0 and r.get("energy_1536", 1.0) <= 0.0]
                below = [r for r in ok if min([t["energy"] for t in r.get("trace", []) if t.get("energy") is not None]
                                              or [0.0]) < base[c]["reference"]["reference_energy_normalized"] - 1e-6]
                errs = np.array([r["relative_H1_error"] for r in ok])
                ens = np.array([r["energy_1536"] for r in ok])
                col = [r["collision"]["collided"] for r in ok]
                row[m] = dict(n=len(rr), valid=len(ok),
                              best_error=float(errs.min()) if len(errs) else None,
                              median_error=float(np.median(errs)) if len(errs) else None,
                              best_energy_error=float(errs[int(np.argmin(ens))]) if len(errs) else None,
                              median_fits=float(np.median([r["fits"] for r in ok])) if ok else None,
                              collided=int(sum(col)),
                              zero_gap_runs=int(sum(1 for r in ok if r.get("zero_gap_pairs", 0) > 0)),
                              completions=int(sum(r.get("completions", 0) for r in ok)),
                              stops=sorted({r.get("stop") for r in rr}),
                              breakdowns=len(rr) - len(ok), objective_below_reference=len(below))
            # starts at which C ends at a completed limit: behaviour of NC and S from the same start
            cr = {r["seed"]: r for r in runs if r["case"] == c and r["P"] == P and r.get("label") == "C"
                  and isinstance(r.get("relative_H1_error"), (int, float)) and r["relative_H1_error"] <= 1.0
                  and r.get("energy_1536", 1.0) <= 0.0 and r["collision"]["collided"]}
            def same_start(m):
                return [r for r in runs if r["case"] == c and r["P"] == P and r.get("label") == m and r["seed"] in cr
                        and isinstance(r.get("relative_H1_error"), (int, float)) and r["relative_H1_error"] <= 1.0]
            nc = same_start("NC"); sr = same_start("S")
            hmins = [min(r["final_halfgaps"]) for r in nc if r.get("final_halfgaps")]
            wmax_nc = [r["ordinary_weight_max"] for r in nc if r.get("ordinary_weight_max") is not None]
            wmax_s = [r["ordinary_weight_max"] for r in sr if r.get("ordinary_weight_max") is not None]
            row["collided_starts"] = dict(n=len(cr),
                                          nc_median_min_halfgap=float(np.median(hmins)) if hmins else None,
                                          nc_median_max_weight=float(np.median(wmax_nc)) if wmax_nc else None,
                                          s_median_max_weight=float(np.median(wmax_s)) if wmax_s else None,
                                          error_spread=(float(max(abs(a["relative_H1_error"] - cr[a["seed"]]["relative_H1_error"])
                                                                  for a in nc + sr)) if (nc or sr) else None))
            gal = next(g for g in base[c]["gaussian_galerkin"] if g["P"] == P)
            row["galerkin"] = gal
            best_c = row["C"]["best_error"]
            row["fem_dofs_to_match_C_best"] = fem_dofs_to_match(base[c]["fem_p1"], best_c) if best_c else None
            out[c][P] = row
    return out



def latex_table(summary, runs, base):
    """Rows: case, P, best errors S/NC/C, collided runs S/NC/C, breakdowns, fixed centres, P1 dofs to match C."""
    lines = []
    for c in CASES:
        for P in PS:
            r = summary[c][P]
            best = " & ".join(f"{r[m]['best_error']:.3f}" for m in METHODS)
            col = "/".join(str(r[m]["collided"]) for m in METHODS)
            brk = "/".join(str(r[m]["breakdowns"]) for m in METHODS)
            fem = r["fem_dofs_to_match_C_best"]
            lines.append(f"{'single source' if c == 'single_source' else 'sech$^2$'} & {P} & {best} & {col} & {brk} & "
                         f"{r['galerkin']['relative_H1_error']:.3f} & {('%d' % round(fem)) if fem else '$>512$'}\\\\")
    return "\n".join(lines)


def figure(summary, base, path):
    ps_style.apply_style(7.0)
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 1.95), layout="constrained")
    x = np.array(PS)
    titles = {"single_source": "(a) Single-source forcing", "sech_dipole": r"(b) sech$^2$ forcing"}
    for ax, c in zip(axes[:2], CASES):
        best = [summary[c][P]["C"]["best_error"] for P in PS]
        med = [summary[c][P]["C"]["median_error"] for P in PS]
        ax.plot(x, best, color=ps_style.RED, linestyle="--", marker="^", markersize=3.5, linewidth=1.1,
                label="C, best of 10 starts")
        ax.plot(x, med, color=ps_style.RED, linestyle="none", marker="^", markersize=3.5, markerfacecolor="none",
                markeredgewidth=0.8, label="C, median")
        gal = [summary[c][P]["galerkin"]["relative_H1_error"] for P in PS]
        ax.plot(x, gal, color=ps_style.BLACK, linestyle=":", linewidth=1.1, marker="x", markersize=3.5,
                label="Fixed equispaced centers")
        fem = base[c]["fem_p1"]
        fx = np.array([f["dof"] for f in fem]) / 2.0
        fe = np.array([f["relative_H1_error"] for f in fem])
        sel = (fx >= 2) & (fx <= 16)
        ax.plot(fx[sel], fe[sel], color=ps_style.GRAY, linestyle="-", linewidth=0.9, marker=".",
                label=r"P1 elements, $2P$ unknowns")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(PS); ax.set_xticklabels([str(P) for P in PS])
        ax.set_xlabel(r"Gaussian budget $P$"); ax.set_ylabel(r"Relative $H^1$ error")
        ax.set_title(titles[c]); ax.grid(alpha=0.2, linewidth=0.5, which="both")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        yt = [0.2, 0.3, 0.4, 0.6, 0.9] if c == "single_source" else [0.02, 0.05, 0.1, 0.2, 0.5]
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yt]))
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].legend(frameon=False, fontsize=5.6, loc="lower left")
    ax = axes[2]
    for c, col, mk, lab in (("single_source", ps_style.BLUE, "o", "Single-source"),
                            ("sech_dipole", ps_style.BLACK, "s", r"sech$^2$")):
        frac = [summary[c][P]["C"]["collided"] / max(summary[c][P]["C"]["valid"], 1) for P in PS]
        ax.plot(x, frac, color=col, marker=mk, markersize=3.5, linewidth=1.1, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(PS); ax.set_xticklabels([str(P) for P in PS])
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Gaussian budget $P$"); ax.set_ylabel("Fraction of starts")
    ax.set_title("(c) C ends at a completed limit"); ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    fig.savefig(path, metadata={"Creator": "pstudy-forcing analyze.py; saved records only"})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    runs, base = load()
    summary = summarize(runs, base)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for c in CASES:
        print(c, "reference", {k: round(v, 4) for k, v in base[c]["reference"].items()})
        for P in PS:
            r = summary[c][P]
            print(f"  P={P:2d} " + "  ".join(
                f"{m}: best {r[m]['best_error']:.4f} med {r[m]['median_error']:.4f} coll {r[m]['collided']}/{r[m]['valid']}"
                f" fits {r[m]['median_fits']:.0f}" for m in METHODS)
                + f"  | galerkin {r['galerkin']['relative_H1_error']:.4f}"
                + f"  | FEM dofs to match C-best {r['fem_dofs_to_match_C_best']}")
    (HERE / "table_rows.tex").write_text(latex_table(summary, runs, base) + "\n")
    figure(summary, base, HERE / "figure_budget.pdf")
    files = sorted((HERE / "results").rglob("*.json"))
    (HERE / "inputs.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE)}\n"
                                                for p in files))


if __name__ == "__main__":
    main()
