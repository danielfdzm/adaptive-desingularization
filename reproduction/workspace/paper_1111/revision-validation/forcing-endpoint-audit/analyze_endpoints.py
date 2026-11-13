#!/usr/bin/env python3
"""Re-evaluate saved forcing endpoints and regenerate the qualified budget figure.

No optimization or coefficient refitting is performed. Selection uses the saved
1536-node pointwise energy, over every finite evaluable endpoint, without using
the reference solution. Reference errors are used only to describe outcomes and
the original breakdown convention. All P >= 8 results are exploratory endpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "work/pstudy-forcing-2026-11-12"
sys.path.insert(0, str(STUDY))
import pstudy as p
import numpy as np
import torch

sys.path.insert(0, str(ROOT / "adaptive-desingularization/experiments"))
import plot_style as style
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

CASES = ("single_source", "sech_dipole")
BUDGETS = (2, 4, 6, 8, 12, 16)
METHODS = ("S", "NC", "C")


def finite(x):
    return isinstance(x, (float, int)) and math.isfinite(x)


def retained(r):
    """Original descriptive breakdown convention; never used for selection."""
    return (finite(r.get("relative_H1_error")) and r["relative_H1_error"] <= 1
            and finite(r.get("energy_1536")) and r["energy_1536"] <= 0)


def brief(r):
    return {k: r.get(k) for k in ("seed", "relative_H1_error", "energy_1536",
                                  "final_energy_normalized", "coefficient_norm")}


def checked_energy(r, nodes):
    blocks = copy.deepcopy(r["final_blocks"])
    for b in blocks:
        for key in ("c", "e", "s", "p"):
            if key in b:
                b[key] = np.array(b[key])
    coef = torch.tensor(r["final_coefficients"])
    C, U, O, WS = p.cs.primitives(blocks, 1, 1.0)
    M = WS.T @ p.ck.h1_gram(C, U, O, C, U, O, 1.0) @ WS
    gram_quad = 0.5 * float(coef @ M @ coef)
    x0, w0 = p.gl(768)
    normalization = float(w0 @ p.forcing(r["case"], x0) ** 2)
    out = []
    for n in nodes:
        x, w = p.gl(n)
        X = torch.tensor(x[:, None])
        Phi = p.ck.point_values(X, C, U, O, 1.0) @ WS
        Dphi = (p.ck.point_values(X, C, U, O + 1, 1.0) * U[:, 0]) @ WS
        v, vx = (Phi @ coef).numpy(), (Dphi @ coef).numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            rest = float(w @ (p.CASES[r["case"]]["lam"] * p.remainder(v)
                              - p.forcing(r["case"], x) * v))
            point_quad = 0.5 * float(w @ (v*v + vx*vx))
        out.append(dict(nodes=n, gram_quadratic=gram_quad,
                        pointwise_quadratic=point_quad,
                        gram_energy=(gram_quad + rest) / normalization,
                        pointwise_energy=(point_quad + rest) / normalization))
    return out


def summarize(runs, baselines):
    result = {}
    for case in CASES:
        result[case] = {}
        reference = baselines[case]["reference"]["reference_energy_normalized"]
        for P in BUDGETS:
            row = {}
            for method in METHODS:
                rr = [r for r in runs if (r["case"], r["P"], r.get("label")) == (case, P, method)]
                assert len(rr) == 10, (case, P, method, len(rr))
                ok = [r for r in rr if retained(r)]
                eligible = [r for r in rr if finite(r.get("energy_1536"))]
                selected = min(eligible, key=lambda r: (r["energy_1536"], r["seed"]))
                best = min((r for r in rr if finite(r.get("relative_H1_error"))),
                           key=lambda r: (r["relative_H1_error"], r["seed"]))
                below = []
                for r in rr:
                    energies = [t["energy"] for t in r.get("trace", []) if finite(t.get("energy"))]
                    if finite(r.get("final_energy_normalized")):
                        energies.append(r["final_energy_normalized"])
                    if energies and min(energies) < reference - 1e-6:
                        below.append(r["seed"])
                row[method] = dict(
                    starts=len(rr), retained=len(ok), breakdowns=len(rr)-len(ok),
                    retained_seeds=[r["seed"] for r in ok],
                    breakdown_seeds=[r["seed"] for r in rr if not retained(r)],
                    energy_selected=brief(selected), best_reference_error=brief(best),
                    median_retained_error=statistics.median(r["relative_H1_error"] for r in ok),
                    median_all_errors=statistics.median(r["relative_H1_error"] for r in rr
                                                       if finite(r.get("relative_H1_error"))),
                    collided_retained=sum(bool(r["collision"]["collided"]) for r in ok),
                    below_reference_training_seeds=below,
                    below_reference_final_seeds=[r["seed"] for r in rr
                        if finite(r.get("final_energy_normalized"))
                        and r["final_energy_normalized"] < reference - 1e-6],
                    records=[dict(**brief(r), retained=retained(r),
                                  saved_chart_gradient=r.get("stationarity", {}).get("projected_gradient_max_abs"))
                             for r in rr])
            result[case][P] = row
    return result


def make_figure(summary, baselines):
    style.apply_style(7.0)
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 2.05), layout="constrained")
    x = np.array(BUDGETS)
    for ax in axes:
        ax.axvspan(7.0, 17.5, color="#ECECEC", zorder=0)
        ax.axvline(7.0, color="#BBBBBB", linestyle=":", linewidth=0.6, zorder=0)
        ax.text(0.98, 0.97, "Exploratory", transform=ax.transAxes,
                ha="right", va="top", color=style.GRAY, fontsize=5.6)
    for ax, case, title in zip(axes[:2], CASES, ("(a) Single-source forcing", r"(b) sech$^2$ forcing")):
        rows = [summary[case][P]["C"] for P in BUDGETS]
        ax.plot(x, [r["energy_selected"]["relative_H1_error"] for r in rows],
                color=style.RED, linestyle="--", marker="^", markersize=3.5,
                linewidth=1.1, label="C, energy-selected")
        ax.plot(x, [r["median_retained_error"] for r in rows],
                color=style.RED, linestyle="none", marker="^", markersize=3.5,
                markerfacecolor="none", markeredgewidth=0.8, label="C, retained median")
        for P, r in zip(BUDGETS, rows):
            if r["retained"] < 10:
                ax.annotate(f"n={r['retained']}", (P, r["median_retained_error"]),
                            xytext=(-3, 5), textcoords="offset points", fontsize=5.0,
                            color=style.RED, ha="right")
        gal = [next(g for g in baselines[case]["gaussian_galerkin"] if g["P"] == P)["relative_H1_error"]
               for P in BUDGETS]
        ax.plot(x, gal, color=style.BLACK, linestyle=":", linewidth=1.1,
                marker="x", markersize=3.5, label="Fixed equispaced centers")
        fem = baselines[case]["fem_p1"]
        fx = np.array([f["dof"] for f in fem])/2
        fe = np.array([f["relative_H1_error"] for f in fem])
        use = (fx >= 2) & (fx <= 16)
        ax.plot(fx[use], fe[use], color=style.GRAY, linewidth=0.9, marker=".",
                label=r"P1 elements, $2P$ unknowns")
        ax.set_yscale("log")
        yt = [0.2, 0.3, 0.4, 0.6, 0.9] if case == "single_source" else [0.02, 0.05, 0.1, 0.2, 0.5]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yt]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel(r"Relative $H^1$ error")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=5.25, loc="lower left")
    ax = axes[2]
    for case, color, marker, label in (("single_source", style.BLUE, "o", "Single-source"),
                                       ("sech_dipole", style.BLACK, "s", r"sech$^2$")):
        fraction = [summary[case][P]["C"]["collided_retained"] / summary[case][P]["C"]["retained"]
                    for P in BUDGETS]
        ax.plot(x, fraction, color=color, marker=marker, markersize=3.5, linewidth=1.1, label=label)
    ax.set_ylim(-0.04, 1.14)
    ax.set_ylabel("Fraction of retained runs")
    ax.set_title("(c) Completed limits in C")
    ax.legend(frameon=False, fontsize=5.8, loc="center right")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(1.8, 17.5)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(P) for P in BUDGETS])
        ax.set_xlabel(r"Gaussian budget $P$")
        ax.grid(alpha=0.2, linewidth=0.5, which="both")
    fig.savefig(HERE / "figure_budget.pdf", metadata={"Creator": "analyze_endpoints.py; saved endpoints, no optimization rerun"})
    fig.savefig(HERE / "figure_budget.png", dpi=250)
    plt.close(fig)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    paths = sorted((STUDY / "results/runs").glob("*.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    baselines = {case: json.loads((STUDY / f"results/baselines/{case}.json").read_text()) for case in CASES}
    summary = summarize(runs, baselines)
    diagnostics = []
    for r in runs:
        if r["P"] >= 8:
            item = dict(case=r["case"], P=r["P"], method=r["label"], seed=r["seed"],
                        retained=retained(r), saved_pointwise_energy=r.get("energy_1536"))
            special = (r["case"], r["P"], r["label"], r["seed"]) in {
                ("sech_dipole", 16, "C", 308), ("single_source", 16, "C", 306)}
            try:
                item["evaluations"] = checked_energy(r, (384, 768, 1536, 3072) if special else (768, 1536))
            except Exception as exc:
                item["evaluation_error"] = str(exc)
            diagnostics.append(item)
    data = dict(scope="Saved endpoint audit only; no optimization or coefficient refits. P>=8 is exploratory.",
                selection="Minimum finite saved pointwise energy_1536 over all ten starts; no reference-error filter.",
                retained="Original descriptive breakdown convention: finite error <=1 and finite pointwise energy <=0.",
                summary=summary, high_budget_endpoint_checks=diagnostics)
    (HERE / "summary.json").write_text(json.dumps(clean(data), indent=2, allow_nan=False)+"\n")
    files = paths + [STUDY / f"results/baselines/{case}.json" for case in CASES]
    (HERE / "inputs.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n" for path in files))
    make_figure(summary, baselines)
    for case in CASES:
        for P in BUDGETS:
            r = summary[case][P]
            print(case, P, "selected S/NC/C", *(f"{r[m]['energy_selected']['relative_H1_error']:.6f}" for m in METHODS),
                  "retained", *(r[m]["retained"] for m in METHODS))
    checks = [r for r in diagnostics if "evaluations" in r]
    differences = [abs(next(v for v in r["evaluations"] if v["nodes"] == 768)["pointwise_energy"]
                       - next(v for v in r["evaluations"] if v["nodes"] == 1536)["pointwise_energy"])
                   for r in checks if r["retained"]]
    print("High-budget endpoint checks:", len(checks), "of", len(diagnostics))
    print("Maximum retained pointwise energy change, 768 to 1536:", max(differences))


if __name__ == "__main__":
    main()
