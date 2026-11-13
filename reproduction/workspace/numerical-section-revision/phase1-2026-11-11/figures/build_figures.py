#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): the paper's numerical figures with C = revised Algorithm 1.

C means pairs in t = h^2 >= 0 (chain solver method "tpair"; T records of the first-order pair solver;
variance_t "C" records), with discrete completion of joined blocks.  S and NC are the records used before.
Only saved records are read; nothing is optimized.  Usage:
    python3.14 build_figures.py [--only order recovery dimension variance solution1d solutionpb]
Writes figure_*.pdf/png, caption_facts.json and provenance.json next to this script.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
NSR = P1.parent
ROOT = NSR.parent
G = NSR / "guarded-rerun-2026-11-09"
SQ = NSR / "sinum-revision-2026-11-10" / "squared-gap" / "results"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phase1-figures-mpl"))
sys.path.insert(0, str(ROOT / "adaptive-desingularization" / "experiments"))
import plot_style                                            # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import LogNorm, Normalize             # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
from matplotlib.patches import ConnectionPatch               # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, NullFormatter,  # noqa: E402
                               NullLocator, ScalarFormatter)
import matplotlib.patheffects as pe                          # noqa: E402
import mpmath as mp                                          # noqa: E402
import numpy as np                                           # noqa: E402

COLORS = {"S": plot_style.BLUE, "NC": plot_style.GREEN, "C": plot_style.RED}
STYLES = {"S": "-", "NC": "-.", "C": "--"}
LABELS = {"S": "Ordinary (S)", "NC": "No completion (NC)", "C": "Completion (C)"}
INPUTS, FACTS = {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def reads(pattern):
    return [read(p) for p in sorted(glob.glob(str(pattern)))]


def decorate(ax):
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", labelsize=8.8)
    ax.tick_params(which="minor", length=1.3)


def log_fits_axis(ax, hi=1850):
    ax.set_xscale("log")
    ax.set_xlim(.85, hi)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, stem, title):
    out = {}
    for ext in ("pdf", "png"):
        path = HERE / f"{stem}.{ext}"
        meta = {"Creator": "Phase 1 figure builder", "Title": title}
        if ext == "pdf":
            meta.update(CreationDate=None, ModDate=None)
        fig.savefig(path, dpi=300 if ext == "png" else 600, bbox_inches="tight", pad_inches=.035, metadata=meta)
        out[ext] = sha(path)
    plt.close(fig)
    return out


def trace_xy(run, key):
    x = np.array([t["fits"] for t in run["trace"]], float)
    y = np.array([t[key] for t in run["trace"]], float)
    return x, y


def end_marker(ax, run, x_end, y_end, method, threshold=1e-6):
    if run.get("stop") == "work_limit":
        ax.plot(x_end, y_end, marker=">", ms=3.8, mfc="white", mec=COLORS[method], mew=.7, ls="none", zorder=6)
    elif run["final_relative_error"] > threshold:
        ax.plot(x_end, y_end, marker="x", ms=3.7, color=COLORS[method], mew=.7, ls="none", zorder=6)


# ------------------------------------------------------------------ Figure: higher order (figure_order)
def figure_order():
    runs = {"S": reads(G / "historical-baselines/higher_d5_*_standard.json"),
            "NC": reads(G / "directional/accuracy-higher/higher_*_stable.json"),
            "C": reads(P1 / "runs/higher/higher_*_tpair.json")}
    expected = {(g, s) for g in (0, 1) for s in range(10, 15)}
    missing = sorted(expected - {(r["geometry"], r["seed"]) for r in runs["C"]})
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.76, wspace=.38)
    extrema = [[], []]
    for m in ("S", "NC", "C"):
        for r in runs[m]:
            for j, key in enumerate(("relative_error", "coefficient_norm")):
                x, y = trace_xy(r, key)
                ok = np.isfinite(y) & (y > 0)
                axes[j].plot(x[ok], y[ok], color=COLORS[m], ls=STYLES[m], lw=.8, alpha=.68)
                end = r["final_relative_error"] if j == 0 else r["coefficient_norm"]
                end_marker(axes[j], r, r["fits"], end, m)
                if r["stop"] != "work_limit" and r["final_relative_error"] <= 1e-6:
                    axes[j].plot(r["fits"], end, marker=".", color=COLORS[m], ms=3.5, ls="none")
                extrema[j].extend(y[ok]); extrema[j].append(end)
                if m == "C":
                    counts = np.array([t["completions"] for t in r["trace"]])
                    idx = np.flatnonzero(np.diff(counts, prepend=0) > 0)
                    idx = idx[y[idx] > 0]
                    axes[j].plot(x[idx], y[idx], marker="o", ms=3, mfc="white", mec=COLORS[m], mew=.6, ls="none")
    target = float(np.linalg.norm([1.1, -.8, 1.4, -1.2, .9, -.7] +
                                  [.45 ** r * b for r, b in zip([2, 2, 3, 3], [-1.3, 1.5, .9, -1.1])]))
    axes[1].axhline(target, color=".45", ls=":", lw=.8)
    for j, (ax, title) in enumerate(zip(axes, ("(a) Accuracy", "(b) Stored coefficients"))):
        ax.set_yscale("log")
        log_fits_axis(ax, 3600)
        lo, hi = min(extrema[j]), max(extrema[j])
        ax.set_ylim(max(lo * .55, 1e-16), hi * 1.5)
        ax.set_title(title)
        decorate(ax)
    axes[0].set_ylabel(r"Relative $H^1$ error")
    axes[1].set_ylabel(r"Stored coefficient $\ell^2$ norm")
    fig.supxlabel("Coefficient fits (log scale)", x=.55, y=.04, fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=1.2, label=LABELS[m]) for m in COLORS]
    handles += [Line2D([0], [0], color=".2", marker="o", mfc="white", ls="none", ms=3.5, label="Block completion"),
                Line2D([0], [0], color=".2", marker=">", ls="none", ms=3.5, label="Fit limit"),
                Line2D([0], [0], color=".2", marker="x", ls="none", ms=3.5, label=r"Return with $E>10^{-6}$")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.53, 1.01), columnspacing=1.1)
    out = save(fig, "figure_order", "Higher-order recovery, d = 5: S, NC and the revised C")
    E = {m: [r["final_relative_error"] for r in runs[m]] for m in runs}
    FACTS["figure_order"] = {
        "runs": {m: len(v) for m, v in runs.items()}, "missing_C_runs_geometry_seed": missing,
        "successes_1e6": {m: sum(e <= 1e-6 for e in v) for m, v in E.items()},
        "median_error": {m: float(np.median(v)) for m, v in E.items()},
        "S_all_at_fit_limit": all(r["stop"] == "work_limit" for r in runs["S"]),
        "C_completions_per_run": [r["completions"] for r in runs["C"]],
        "completed_target_norm": target, "outputs": out}


# ------------------------------------------------------------------ Figure: recovery progress (figure_recovery)
def figure_recovery():
    lin = {"NC": reads(P1 / "runs/linear1d/linear1d_*_stable.json"),
           "C": reads(P1 / "runs/linear1d/linear1d_*_tpair.json")}
    pb = {"NC": reads(G / "first-order/accuracy/pb_d10_*_stable.json"),
          "C": reads(SQ / "pb_d10_*_squared_gap.json")}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.65))
    fig.subplots_adjust(left=.105, right=.985, bottom=.22, top=.76, wspace=.36)
    t0_marks = 0
    for ax, runs in zip(axes, (lin, pb)):
        for m in ("NC", "C"):
            for r in runs[m]:
                x, y = trace_xy(r, "relative_error")
                ok = y > 0
                ax.plot(x[ok], y[ok], color=COLORS[m], ls=STYLES[m], lw=.85, alpha=.68)
                abnormal = bool(r.get("abnormal_returns", 0)) or r.get("stop") == "abnormal"
                if abnormal:
                    ax.plot(r["fits"], r["final_relative_error"], marker="x", ms=3.7, color=COLORS[m], mew=.7, ls="none", zorder=6)
                if r.get("stop") == "work_limit":
                    ax.plot(r["fits"], r["final_relative_error"], marker=">", ms=3.8, mfc="white", mec=COLORS[m], mew=.7, ls="none", zorder=6)
                ax.plot(r["fits"], r["final_relative_error"], marker=".", ms=3.2, color=COLORS[m], ls="none", zorder=5)
                if m == "C":
                    counts = np.array([t.get("completions", 0) for t in r["trace"]])
                    idx = np.flatnonzero(np.diff(counts, prepend=0) > 0)
                    idx = idx[y[idx] > 0]
                    ax.plot(x[idx], y[idx], marker="o", ms=2.7, mfc="white", mec=COLORS[m], mew=.6, ls="none", zorder=4)
                    zeros = np.array([t.get("zero_squared_gaps", 0) for t in r["trace"]])
                    zidx = np.flatnonzero(np.diff(zeros, prepend=0) > 0)
                    zidx = zidx[y[zidx] > 0]
                    t0_marks += len(zidx)
                    ax.plot(x[zidx], y[zidx], marker="*", ms=4.2, color=COLORS[m], mec="black", mew=.3, ls="none", zorder=5)
        ax.set_yscale("log")
        ax.set_ylabel(r"Relative $H^1$ error", fontsize=10)
        decorate(ax)
    axes[0].set_title(r"(a) Linear, $d=1$", fontsize=10, pad=5)
    axes[1].set_title(r"(b) PB, $d=10$, $\lambda=1$", fontsize=10, pad=5)
    xmax = max(r["fits"] for v in lin.values() for r in v)
    axes[0].set_xlim(-3, 1.07 * xmax)
    axes[0].set_xlabel("Coefficient fits", fontsize=10)
    log_fits_axis(axes[1])
    axes[1].set_xlabel("Coefficient fits (log scale)", fontsize=10)
    for ax, runs in zip(axes, (lin, pb)):
        ys = [v for m in runs for r in runs[m] for v in trace_xy(r, "relative_error")[1] if v > 0]
        ys += [r["final_relative_error"] for m in runs for r in runs[m]]
        ax.set_ylim(.5 * min(ys), 1.6 * max(ys))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=1.3, label=LABELS[m]) for m in ("NC", "C")]
    handles += [Line2D([0], [0], marker=".", ms=5, color="#333333", ls="none", label="Endpoint (80 digits)"),
                Line2D([0], [0], marker="o", ms=4, mfc="white", mec="#333333", ls="none", label="Block completion"),
                Line2D([0], [0], marker="*", ms=5, color="#333333", ls="none", label=r"Pair reaches $t=0$"),
                Line2D([0], [0], marker="x", ms=4, color="#333333", ls="none", label="Abnormal return")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.06), ncol=3, fontsize=8.6,
               columnspacing=1.0, handlelength=1.8)
    out = save(fig, "figure_recovery", "Progress of matched NC and C runs")
    FACTS["figure_recovery"] = {
        "runs": {"linear": {m: len(v) for m, v in lin.items()}, "pb_d10": {m: len(v) for m, v in pb.items()}},
        "linear_median_error": {m: float(np.median([r["final_relative_error"] for r in v])) for m, v in lin.items()},
        "pb_median_error": {m: float(np.median([r["final_relative_error"] for r in v])) for m, v in pb.items()},
        "pb_C_pairs_at_t0_at_end": [r.get("zero_squared_gaps") for r in pb["C"]],
        "t0_markers_drawn": t0_marks,
        "note": "t = 0 markers come from recorded trace rows (zero_squared_gaps); first-order traces are sparse, "
                "so a star marks the first recorded row after a pair reached t = 0.",
        "outputs": out}


# ------------------------------------------------------------------ Figure: dimensions (figure_dimension)
def figure_dimension():
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.55), sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.77, wspace=.18)
    facts = {}
    ys = []
    for ax, d, letter in zip(axes, (5, 10, 20), "abc"):
        runs = {"S": reads(G / f"historical-baselines/pb_d{d}_*_standard.json"),
                "C": reads(SQ / f"pb_d{d}_*_squared_gap.json")}
        for m in ("S", "C"):
            for r in runs[m]:
                x, y = trace_xy(r, "relative_error")
                ax.plot(x, y, color=COLORS[m], ls=STYLES[m], lw=.85, alpha=.68)
                if r.get("stop") == "work_limit":
                    ax.plot(r["fits"], r["final_relative_error"], marker=">", ms=3.8, mfc="white", mec=COLORS[m], mew=.7, ls="none", zorder=6)
                if bool(r.get("abnormal_returns", 0)):
                    ax.plot(r["fits"], r["final_relative_error"], marker="x", ms=3.7, color=COLORS[m], mew=.7, ls="none", zorder=6)
                ys.extend(y[y > 0])
        log_fits_axis(ax)
        ax.set_yscale("log")
        ax.set_title(rf"({letter}) $d={d}$", fontsize=10, pad=5)
        decorate(ax)
        facts[d] = {m: {"n": len(v), "median_error": float(np.median([r["final_relative_error"] for r in v])),
                        "median_fits": float(np.median([r["fits"] for r in v])),
                        "fit_limit": sum(r["stop"] == "work_limit" for r in v)} for m, v in runs.items()}
    axes[0].set_ylim(.5 * min(ys), 1.6 * max(ys))
    axes[0].yaxis.set_major_locator(FixedLocator([1e-1, 1e-3, 1e-5, 1e-7]))
    axes[0].set_ylabel(r"Relative $H^1$ error", fontsize=10)
    fig.supxlabel("Coefficient fits (log scale)", x=.55, y=.035, fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=1.3, label=LABELS[m]) for m in ("S", "C")]
    handles += [Line2D([0], [0], marker=">", ms=4, mfc="white", mec="#333333", ls="none", label="Coefficient-fit limit")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.015), ncol=3, fontsize=9,
               columnspacing=1.25, handlelength=1.8)
    out = save(fig, "figure_dimension", "First-order PB recovery in dimensions 5, 10 and 20: S and the revised C")
    FACTS["figure_dimension"] = {"per_dimension": facts, "outputs": out}


# ------------------------------------------------------------------ Figure: variance family (figure_variance_family)
def figure_variance():
    runs = {"S": reads(G / "variance/results/s0.99_seed*_S.json"),
            "NC": reads(G / "variance/results/s0.99_seed*_NC.json"),
            "C": reads(P1 / "runs/variance/s0.99_seed*_C.json")}
    reference = 7.68848e-3
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.05))
    stars = 0
    for m in ("S", "NC", "C"):
        for r in runs[m]:
            it = np.array([p["iteration"] for p in r["trace"]], float) + 1
            for ax, key in zip(axes, ("relative_error", "stored_coefficient_norm")):
                y = np.array([p[key] for p in r["trace"]], float)
                ax.loglog(it, y, color=COLORS[m], ls=STYLES[m], lw=.85, alpha=.65)
            if m == "C":
                hit = next((i for i, p in enumerate(r["trace"]) if p.get("t") == 0.), None)
                if hit is not None:
                    stars += 1
                    p = r["trace"][hit]
                    axes[0].plot(p["iteration"] + 1, p["relative_error"], "*", color=COLORS[m], mec="black", mew=.3, ms=5.5, zorder=5)
                    axes[1].plot(p["iteration"] + 1, p["stored_coefficient_norm"], "*", color=COLORS[m], mec="black", mew=.3, ms=5.5, zorder=5)
    axes[0].axhline(reference, color="#555555", ls=":", lw=1)
    axes[0].set(ylabel=r"Relative $H^1$ error", title="(a) Approximation error")
    axes[1].set(ylabel="Stored coefficient norm", title="(b) Coefficient growth")
    for ax in axes:
        ax.set_xlabel("Optimizer iterations $+1$")
        decorate(ax)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], ls=STYLES[m], label=LABELS[m]) for m in ("S", "NC", "C")]
               + [Line2D([], [], color=COLORS["C"], marker="*", mec="black", mew=.3, ls="none", label=r"Pair reaches $t=0$")],
               loc="upper center", ncol=4, bbox_to_anchor=(.52, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .87))
    out = save(fig, "figure_variance_family", "Gaussian-variance family, s = 0.99: S, NC and the revised C")
    within = {m: sum(abs(r["final"]["relative_error"] - reference) <= .01 * reference for r in v) for m, v in runs.items()}
    FACTS["figure_variance_family"] = {
        "runs": {m: len(v) for m, v in runs.items()}, "within_1pct_of_reference": within,
        "C_runs_reaching_t0": stars,
        "unsuccessful_seeds": {m: [r["seed"] for r in v if abs(r["final"]["relative_error"] - reference) > .01 * reference]
                               for m, v in runs.items()},
        "median_iterations": {m: float(np.median([r["iterations"] for r in v])) for m, v in runs.items()},
        "x_axis": "iterations + 1 (C times are shared-host and are not plotted)", "outputs": out}


# ------------------------------------------------------------------ Figure: 1D solution (figure_solution_1d)
def gaussian_derivative(z, order, sigma):
    y = z / sigma
    h0, h1 = mp.mpf(1), y
    if order == 0:
        hermite = h0
    elif order == 1:
        hermite = h1
    else:
        for n in range(1, order):
            h0, h1 = h1, y * h1 - n * h0
        hermite = h1
    return (-1) ** order * hermite * mp.exp(-y * y / 2) / sigma ** order


def chain_state_atoms(rec):
    """Primitives (center, x-derivative order, weight) of a 1D chain-solver state, via its own primitives."""
    sys.path.insert(0, str(P1 / "src"))
    import chain_solver as cs
    blocks = []
    for b in rec["final_blocks"]:
        nb = {k: (np.asarray(v, float) if k in ("c", "e", "s", "p") else v) for k, v in b.items()}
        if "p" in nb:
            nb["p"] = nb["p"].reshape(-1, 1)
        blocks.append(nb)
    C, U, O, WS = cs.primitives(blocks, 1, 0.6)
    w = (WS @ __import__("torch").as_tensor(np.asarray(rec["final_coefficients"], float))).numpy()
    atoms = []
    for i in range(len(O)):
        u = float(U[i][0]); o = int(O[i])
        atoms.append((float(C[i][0]), o, float(w[i]) * (u ** o)))      # d_u^o G = u^o G^(o) in 1D
    return blocks, atoms


def figure_solution1d():
    S_all = reads(P1 / "runs/linear1d/linear1d_*_standard_restart.json")
    # Rule (fixed 11 Nov 2026, after the trial amendment of the solver): the first start in which S ends with
    # relative error below 1e-4, so that the figure shows S near the target with its diverging weights.
    good = [r for r in S_all if r["final_relative_error"] < 1e-4]
    seed = min(r["seed"] for r in good)
    S = read(P1 / f"runs/linear1d/linear1d_d1_g0_seed{seed}_standard_restart.json")
    C = read(P1 / f"runs/linear1d/linear1d_d1_g0_seed{seed}_tpair.json")
    sigma = mp.mpf("0.6")
    T = dict(centers=[0.0, 0.0, 2.2, 2.2, 2.2, -2.1], orders=[0, 1, 0, 1, 2, 0], coeffs=[0.8, -1.2, -0.9, 0.6, 0.5, 1.0])
    target = [(c, r, a) for c, r, a in zip(T["centers"], T["orders"], T["coeffs"])]
    _, S_atoms = chain_state_atoms(S)
    C_blocks, C_atoms = chain_state_atoms(C)
    x = np.linspace(-4.5, 4.5, 901)
    fields = {"target": [], "S": [], "C": []}
    errors = {"S": [], "C": []}
    with mp.workdps(40):
        for xi in x:
            xm = mp.mpf(float(xi))
            val = {k: mp.fsum(mp.mpf(a) * gaussian_derivative(xm - mp.mpf(c), r, sigma) for c, r, a in atoms)
                   for k, atoms in (("target", target), ("S", S_atoms), ("C", C_atoms))}
            for k in fields:
                fields[k].append(float(val[k]))
            for k in errors:
                errors[k].append(float(abs(val[k] - val["target"])))
    fields = {k: np.array(v) for k, v in fields.items()}
    errors = {k: np.maximum(np.array(v), 1e-16) for k, v in errors.items()}
    # representation per target center
    S_c = np.array([b["c"][0] for b in S["final_blocks"]]); S_w = np.array(S["final_coefficients"])
    plot_style.apply_style(9.5)
    fig = plt.figure(figsize=(6.25, 4.7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.6, 1.45], hspace=.78, wspace=.55)
    upper = gs[0, :].subgridspec(1, 2, wspace=.28)
    ax_t = fig.add_subplot(upper[0, 0]); ax_e = fig.add_subplot(upper[0, 1])
    zooms = [fig.add_subplot(gs[1, j * 2:(j + 1) * 2]) for j in range(3)]
    fig.subplots_adjust(left=.105, right=.985, top=.86, bottom=.13)
    ax_t.plot(x, fields["target"], c=plot_style.BLACK, lw=1.5)
    ax_t.set(xlim=(-4.5, 4.5), xticks=[-4, -2.1, 0, 2.2, 4], xlabel="$x$", ylabel=r"$u^\ast(x)$")
    ax_t.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    ax_t.set_title("(a) Target")
    ax_e.semilogy(x, errors["S"], c=COLORS["S"], ls="-", lw=1.05)
    ax_e.semilogy(x, errors["C"], c=COLORS["C"], ls="--", lw=1.05)
    ax_e.set(xlim=(-4.5, 4.5), xticks=[-4, -2, 0, 2, 4], xlabel="$x$", ylabel=r"$|v(x)-u^\ast(x)|$")
    lo = max(min(errors["C"].min(), errors["S"].min()), 1e-15)
    ax_e.set_ylim(lo / 3, 3 * max(errors["S"].max(), errors["C"].max()))
    ax_e.set_title("(b) Pointwise error")
    linthresh = 1.
    allvals = list(S_w)
    centers_info = []
    for b, g in zip(C_blocks, np.split(np.asarray(C["final_coefficients"]), np.cumsum([1 if b["kind"] == "atom" else 2 if b["kind"] == "pair" else b["k"] + 1 for b in C_blocks])[:-1])):
        e = float(b["e"][0]) if "e" in b else 1.0
        if b["kind"] == "atom":
            coefs = [float(g[0])]
        elif b["kind"] == "pair":
            coefs = [float(g[0]), float(-e * g[1])]                       # limits G and -d_e G = -e G'
        else:
            coefs = [float(g[r]) * (-e) ** r for r in range(b["k"] + 1)]  # (-1)^r d_e^r G = (-e)^r G^(r)
        allvals += coefs
        centers_info.append((b, coefs))
    top = 10 ** math.ceil(math.log10(max(abs(v) for v in allvals))) * 50
    bottom = -10 ** math.ceil(math.log10(max(abs(v) for v in allvals)))
    ticks = {v: lab for v, lab in [(-1e4, r"$-10^4$"), (-1e2, r"$-10^2$"), (-1., r"$-1$"), (1., r"$1$"), (1e2, r"$10^2$"), (1e4, r"$10^4$")]
             if bottom < v < top}
    z_facts = []
    for label, ax, c, hw, m in zip(["i", "ii", "iii"], zooms, [-2.1, 0., 2.2], [3e-7, .02, .02], [1, 2, 3]):
        near = np.abs(S_c - c) < .6
        offs, ws = S_c[near] - c, S_w[near]
        blk = min(centers_info, key=lambda bc: abs(float(bc[0]["c"][0]) - c))
        b, coefs = blk
        bc = float(b["c"][0]) - c
        hw = max(hw, 1.25 * float(np.max(np.abs(offs))) if len(offs) else hw, 1.25 * abs(bc))
        ax.set_facecolor("#fafbfd")
        ax.set_yscale("symlog", linthresh=linthresh, linscale=.9)
        ax.axhline(0, c="#a5adb6", lw=.6, zorder=1); ax.axvline(0, c="#a5adb6", ls=":", lw=.75, zorder=1)
        ax.vlines(offs, 0, ws, color=COLORS["S"], lw=.95, zorder=2)
        ax.plot(offs, ws, "o", c=COLORS["S"], ms=4, mew=.6, mfc="white", ls="none", zorder=3)
        ax.vlines(np.full(len(coefs), bc), 0, coefs, color=COLORS["C"], lw=1.1, zorder=4)
        marker, size = {"atom": ("s", 5), "pair": ("D", 4.5), "chain": ("*", 7.5)}[b["kind"]]
        ax.plot(np.full(len(coefs), bc), coefs, marker=marker, c=COLORS["C"], ms=size, mew=.55, mec="black", ls="none", zorder=5)
        if b["kind"] == "pair":
            h = math.sqrt(max(float(b["t"]), 0.))
            mant, ex = f"{float(b['t']):.1e}".split("e")
            ax.text(.97, .06, rf"$t={mant}\times10^{{{int(ex)}}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2)
        ax.set(xlim=(-hw, hw), ylim=(bottom, top))
        ax.yaxis.set_major_locator(FixedLocator(list(ticks))); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ticks.get(v, "")))
        if ax is not zooms[0]:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel(r"$c-c^\ast$", labelpad=3)
        ax.set_title(rf"({label}) $c^\ast={c:g}$, $m={m}$".replace("-", "−"), fontsize=8.6, y=.96, pad=0, va="top")
        fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(FixedLocator([-round(hw * .7, 12), 0, round(hw * .7, 12)]))
        for spine in ax.spines.values():
            spine.set_color("#a4adb7"); spine.set_linewidth(.65)
        ax.grid(axis="y", which="major", color="#e0e4e9", lw=.5)
        ax_t.axvspan(c - hw, c + hw, ymin=0, ymax=.075, facecolor="#a5adb6", edgecolor="#8b96a2", alpha=.30, lw=.65, zorder=4)
        ax_t.plot([c], [0], marker="o", ms=3.4, mfc="white", mec="#7e8995", mew=.7, ls="none",
                  transform=ax_t.get_xaxis_transform(), clip_on=False, zorder=6)
        for sx, dx in [(c - hw, 0.), (c + hw, 1.)]:
            fig.add_artist(ConnectionPatch(xyA=(sx, 0.), coordsA=ax_t.get_xaxis_transform(), xyB=(dx, 1.),
                                           coordsB=ax.transAxes, axesA=ax_t, axesB=ax, arrowstyle="-",
                                           color="#a4adb7", lw=.55, alpha=.75, clip_on=False, zorder=-.5))
        z_facts.append({"target_center": c, "S_offsets": offs.tolist(), "S_weights": ws.tolist(),
                        "C_block": b["kind"], "C_center_offset": bc, "C_coefficients_of_G_derivatives": coefs,
                        "C_t": float(b.get("t", float("nan"))) if b["kind"] == "pair" else None,
                        "half_width": hw})
    zooms[0].set_ylabel("Coefficient", labelpad=2)
    for ax in (ax_t, ax_e):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", which="major", color="#dddddd", lw=.45)
    for lab in ax_t.get_xticklabels() + [ax_t.xaxis.label]:
        lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=.45))
    fig.legend(handles=[Line2D([], [], c=plot_style.BLACK, lw=1.5, label="Target"),
                        Line2D([], [], c=COLORS["S"], lw=1.2, marker="o", ms=4, mfc="white", mew=.6, label=LABELS["S"]),
                        Line2D([], [], c=COLORS["C"], lw=1.2, ls="--", marker="*", ms=7, mec="black", mew=.5, label=LABELS["C"])],
               loc="upper center", bbox_to_anchor=(.55, .995), ncol=3, fontsize=9)
    fig.text(.545, .02, r"S: Gaussian weights. C: coefficients of $G, G', G''$ in each block "
             r"(square: singleton, diamond: pair with $t>0$, star: completed block).", ha="center", va="center", fontsize=8.1)
    out = save(fig, "figure_solution_1d", "Linear recovery: target, pointwise errors and final representations")
    FACTS["figure_solution_1d"] = {
        "seed": seed, "seed_rule": "first seed in 0-9 whose S run ends with relative error below 1e-4",
        "S_final_error": S["final_relative_error"], "C_final_error": C["final_relative_error"],
        "S_max_abs_weight": float(np.max(np.abs(S_w))), "C_blocks": [b["kind"] for b in C_blocks],
        "zoom_panels": z_facts, "max_pointwise_error": {k: float(v.max()) for k, v in errors.items()},
        "outputs": out}


# ------------------------------------------------------------------ Figure: PB solution (figure_solution_pb)
SIG = 0.45
LD = np.longdouble


def pb_terms(blocks, coef, pts):
    pts = np.asarray(pts, dtype=LD)
    s2 = LD(SIG) ** 2
    cols, k = [], 0
    for b in blocks:
        off = pts - np.array(b["c"], dtype=LD)
        r = np.sum(off * off, axis=1)
        if b["kind"] == "single":
            cols.append(LD(coef[k]) * np.exp(-r / (2 * s2))); k += 1
            continue
        e = np.array(b["e"], dtype=LD)
        s = np.sum(off * e, axis=1) / s2
        t = LD(0.) if b.get("completed", False) else LD(b.get("t", b.get("h", 0.) ** 2))
        a = np.exp(-(r + t) / (2 * s2))
        y = s * np.sqrt(t)
        small = np.abs(y) < LD(1e-4)
        ys = np.where(small, LD(1), y)
        sinhc = np.where(small, 1 + y * y / 6 + y ** 4 / 120, np.sinh(ys) / ys)
        cols.append(LD(coef[k]) * a * np.cosh(y))
        cols.append(LD(coef[k + 1]) * a * s * sinhc)
        k += 2
    assert k == len(coef)
    return np.array(cols)


def neumaier(cols):
    total = np.zeros(cols.shape[1], dtype=LD); corr = np.zeros_like(total)
    for x in cols:
        up = total + x
        corr += np.where(abs(total) >= abs(x), (total - up) + x, (x - up) + total)
        total = up
    return total + corr


def pb_mp_value(blocks, coef, point):
    s2 = mp.mpf(SIG) ** 2
    out, k = [], 0
    for b in blocks:
        off = [mp.mpf(float(x)) - mp.mpf(float(c)) for x, c in zip(point, b["c"])]
        r = mp.fsum(v * v for v in off)
        if b["kind"] == "single":
            out.append(mp.mpf(coef[k]) * mp.exp(-r / (2 * s2))); k += 1
            continue
        s = mp.fsum(v * mp.mpf(e) for v, e in zip(off, b["e"])) / s2
        t = mp.mpf(0) if b.get("completed", False) else mp.mpf(b.get("t", b.get("h", 0.) ** 2))
        a = mp.exp(-(r + t) / (2 * s2))
        y = s * mp.sqrt(t)
        out.append(mp.mpf(coef[k]) * a * mp.cosh(y))
        out.append(mp.mpf(coef[k + 1]) * a * s * (mp.sinh(y) / y if y != 0 else mp.mpf(1)))
        k += 2
    return mp.fsum(out)


def figure_solutionpb():
    S = read(G / "historical-baselines/pb_d10_g0_s10_standard.json")
    C = read(SQ / "pb_d10_g0_s10_squared_gap.json")
    for name in ("initial_centers", "target_blocks", "target_coefficients"):
        assert S[name] == C[name], name
    runs = {"S": S, "C": C}
    tb, tc = S["target_blocks"], S["target_coefficients"]
    cen = np.array([b["c"] for b in tb], float)
    origin = cen.mean(axis=0)
    _, sv, right = np.linalg.svd(cen - origin, full_matrices=False)
    basis = right[:2].copy()
    for ax_ in basis:
        if ax_[np.argmax(abs(ax_))] < 0:
            ax_ *= -1
    project = lambda c: (np.asarray(c) - origin) @ basis.T
    grid = np.linspace(-1.3, 1.3, 151)
    s1, s2 = np.meshgrid(grid, grid, indexing="xy")
    pts = origin + s1.ravel()[:, None] * basis[0] + s2.ravel()[:, None] * basis[1]
    tterms = pb_terms(tb, tc, pts)
    fields = {"target": neumaier(tterms)}
    errors = {}
    for m, r in runs.items():
        mt = pb_terms(r["blocks"], r["coefficients"], pts)
        fields[m] = neumaier(mt)
        errors[m] = abs(neumaier(np.concatenate([mt, -tterms])))
    # 80-digit spot validation
    idx = sorted(set(np.linspace(0, len(pts) - 1, 15, dtype=int).tolist() + [int(np.argmax(errors["C"])), int(np.argmax(errors["S"]))]))
    worst = 0.
    with mp.workdps(60):
        for i in idx:
            ex_t = pb_mp_value(tb, tc, pts[i])
            for m, r in runs.items():
                ex = pb_mp_value(r["blocks"], r["coefficients"], pts[i])
                worst = max(worst, float(abs(mp.mpf(float(fields[m][i])) - ex)), float(abs(mp.mpf(float(errors[m][i])) - abs(ex - ex_t))))
    assert worst < 3e-13, worst
    vlim = float(max(np.max(abs(v)) for v in fields.values()))
    efloor = 1e-10
    eup = 10.0 ** math.ceil(math.log10(max(float(v.max()) for v in errors.values())))
    plot_style.apply_style(font_size=8.8)
    fig = plt.figure(figsize=(6.3, 4.85))
    axes = [fig.add_axes((left, bottom, .2297, .2984)) for bottom in (.65, .20) for left in (.08, .39, .70)]
    half = float(grid[1] - grid[0]) / 2
    extent = [float(grid[0] - half), float(grid[-1] + half)] * 2
    fnorm, enorm = Normalize(-vlim, vlim), LogNorm(efloor, eup, clip=True)
    proj_t = project([b["c"] for b in tb])
    proj_S = project([b["c"] for b in S["blocks"]])
    singles = [b for b in C["blocks"] if b["kind"] == "single"]
    pairs = [b for b in C["blocks"] if b["kind"] == "pair"]
    zero = [b for b in pairs if b.get("t", 1.) == 0.]
    open_ = [b for b in pairs if b.get("t", 1.) != 0.]
    for i, (name, title) in enumerate((("target", "(a) Target"), ("S", "(b) Ordinary (S)"), ("C", "(c) Completion (C)"))):
        ax = axes[i]
        img = ax.imshow(np.asarray(fields[name], float).reshape(s1.shape), origin="lower", extent=extent, cmap="RdBu_r",
                        norm=fnorm, interpolation="nearest")
        ax.set_title(title)
        if name == "target":
            ax.scatter(*proj_t.T, marker="x", s=25, c="0.12", linewidths=.9, zorder=5,
                       path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        elif name == "S":
            ax.scatter(*proj_S.T, marker="o", s=25, facecolors="none", edgecolors=COLORS["S"], linewidths=1.0, zorder=6,
                       path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        else:
            if singles:
                ax.scatter(*project([b["c"] for b in singles]).T, marker="^", s=25, c=COLORS["C"], edgecolors="white", linewidths=.5, zorder=7)
            if zero:
                ax.scatter(*project([b["c"] for b in zero]).T, marker="*", s=45, c=COLORS["C"], edgecolors="white", linewidths=.5, zorder=7)
            if open_:
                ax.scatter(*project([b["c"] for b in open_]).T, marker="D", s=20, c=COLORS["C"], edgecolors="white", linewidths=.5, zorder=7)
        if name in runs:
            mant, ex = f"{runs[name]['final_relative_error']:.2e}".split("e")
            ax.text(.03, .03, rf"$\varepsilon_H={mant}\!\times\!10^{{{int(ex)}}}$", transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.2))
    for i, m in enumerate(("S", "C"), start=3):
        eimg = axes[i].imshow(np.maximum(np.asarray(errors[m], float), efloor).reshape(s1.shape), origin="lower", extent=extent,
                              cmap="magma", norm=enorm, interpolation="nearest")
        axes[i].set_title(f"({'d' if m == 'S' else 'e'}) Error: {m}")
    for i, ax in enumerate(axes[:5]):
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        if i < 3:
            ax.tick_params(labelbottom=False)
        if i in (0, 3):
            ax.set_ylabel(r"$s_2$")
        if i in (3, 4):
            ax.set_xlabel(r"$s_1$")
    # zoom on the first target cluster
    ax = axes[5]
    ax.set_title("(f) First cluster: zoom")
    zc = proj_t[0]
    tcenter = np.array(tb[0]["c"], float)
    S_near = sorted(range(len(S["blocks"])), key=lambda j: np.linalg.norm(np.array(S["blocks"][j]["c"]) - tcenter))[:2]
    C_pair = min(pairs, key=lambda b: np.linalg.norm(np.array(b["c"]) - tcenter))
    init = project(S["initial_centers"])
    init_near = sorted(range(len(init)), key=lambda j: np.linalg.norm(np.array(S["initial_centers"][j]) - tcenter))[:2]
    hw = .045
    ax.scatter(*(init[init_near] - zc).T, marker=".", s=24, c="0.55", zorder=4)
    ax.scatter(*(proj_t[:1] - zc).T, marker="x", s=65, c="0.12", linewidths=.9, zorder=5,
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.scatter(*(proj_S[S_near] - zc).T, marker="o", s=45, facecolors="none", edgecolors=COLORS["S"], linewidths=1.0, zorder=6)
    pc = project([C_pair["c"]])[0] - zc
    mk = "*" if C_pair.get("t", 1.) == 0. else "D"
    ax.scatter([pc[0]], [pc[1]], marker=mk, s=60 if mk == "*" else 30, c=COLORS["C"], edgecolors="white", linewidths=.5, zorder=7)
    ax.axhline(0, color=".9", lw=.5, zorder=0); ax.axvline(0, color=".9", lw=.5, zorder=0)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw); ax.set_xticks([-.04, 0, .04]); ax.set_yticks([-.04, 0, .04]); ax.set_aspect("equal")
    ax.set_xlabel(r"$s_1-s_{1,0}$"); ax.set_ylabel(r"$s_2-s_{2,0}$", labelpad=0); ax.tick_params(labelsize=6.9, pad=1)
    vb = fig.colorbar(img, cax=fig.add_axes((.08, .600, .8497, .016)), orientation="horizontal")
    vb.set_label("Solution value on the slice (shared scale)", fontsize=8, labelpad=0); vb.ax.tick_params(labelsize=7, length=2, pad=1)
    eb = fig.colorbar(eimg, cax=fig.add_axes((.08, .100, .8497, .016)), orientation="horizontal", ticks=[1e-10, 1e-8, 1e-6, 1e-4])
    eb.set_label(r"$|v-u^*|$ on the slice (shared log scale)", fontsize=8, labelpad=0); eb.ax.tick_params(labelsize=7, length=2, pad=1)
    eb.ax.minorticks_off()
    handles = [Line2D([], [], ls="none", marker="x", color=".12", label="Target center", ms=5),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=COLORS["S"], label=LABELS["S"], ms=5),
               Line2D([], [], ls="none", marker="*", color=COLORS["C"], label=r"Pair, $t=0$", ms=7),
               Line2D([], [], ls="none", marker="D", color=COLORS["C"], label=r"Pair, $t>0$", ms=4),
               Line2D([], [], ls="none", marker="^", color=COLORS["C"], label="Singleton", ms=5),
               Line2D([], [], ls="none", marker=".", color=".55", label="Initial", ms=5)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, -.015), ncol=6, fontsize=7.4,
               handletextpad=.3, columnspacing=.8, handlelength=1)
    out = save(fig, "figure_solution_pb", "PB d = 10, geometry 0, seed 10: S and the revised C on a plane through the target")
    FACTS["figure_solution_pb"] = {
        "S_error": S["final_relative_error"], "C_error": C["final_relative_error"],
        "C_pairs": len(pairs), "C_pairs_at_t0": len(zero), "C_squared_gaps": [b.get("t") for b in pairs],
        "C_singletons": len(singles), "validation_80digit_max_abs_discrepancy": worst,
        "slice_max_abs_error": {m: float(v.max()) for m, v in errors.items()},
        "error_floor_display": efloor, "outputs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["order", "recovery", "dimension", "variance", "solution1d", "solutionpb"])
    args = ap.parse_args()
    plot_style.apply_style(10)
    jobs = {"order": figure_order, "recovery": figure_recovery, "dimension": figure_dimension,
            "variance": figure_variance, "solution1d": figure_solution1d, "solutionpb": figure_solutionpb}
    old = json.loads((HERE / "caption_facts.json").read_text()) if (HERE / "caption_facts.json").exists() else {}
    for name in args.only:
        plot_style.apply_style(10)
        jobs[name]()
    old.update(FACTS)
    (HERE / "caption_facts.json").write_text(json.dumps(old, indent=1, default=float) + "\n")
    prov = json.loads((HERE / "provenance.json").read_text()) if (HERE / "provenance.json").exists() else {"inputs": {}}
    prov["inputs"].update(INPUTS)
    prov["generator_sha256"] = sha(Path(__file__))
    prov["plot_style_sha256"] = sha(ROOT / "adaptive-desingularization/experiments/plot_style.py")
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(FACTS, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
