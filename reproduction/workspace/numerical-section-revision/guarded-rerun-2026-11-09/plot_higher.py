#!/usr/bin/env python3
"""Build Figure 4 from all guarded NC/C runs and unchanged historical S."""
import os
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "guarded-paper-mpl"))
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
COLORS = {"standard": "#0000FF", "stable": "#188038", "adaptive": "#FF0000"}
STYLES = {"standard": "-", "stable": "-.", "adaptive": "--"}
LABELS = {"standard": "Ordinary (S)", "stable": "No completion (NC)", "adaptive": "Completion (C)"}


def main():
    paths = sorted((HERE / "historical-baselines").glob("higher*_standard.json"))
    paths += sorted((HERE / "directional/accuracy-higher").glob("higher*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    for m in COLORS:
        assert len([r for r in rows if r["method"] == m]) == 10, (m, len(rows))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.76, wspace=.38)
    extrema = [[], []]
    for r in rows:
        m = r["method"]
        x = np.array([v["fits"] for v in r["trace"]], float)
        for j, field in enumerate(("relative_error", "coefficient_norm")):
            y = np.array([v[field] for v in r["trace"]], float)
            valid = np.isfinite(y) & (y > 0)
            axes[j].plot(x[valid], y[valid], color=COLORS[m], ls=STYLES[m], lw=.8, alpha=.68)
            end = r["final_relative_error"] if j == 0 else r["coefficient_norm"]
            marker = ">" if r["stop"] == "work_limit" else "x" if r["final_relative_error"] > 1e-6 else "."
            axes[j].plot(r["fits"], end, marker=marker, color=COLORS[m], ms=3.5, mew=.7, ls="none")
            extrema[j].extend(y[valid]); extrema[j].append(end)
            if m == "adaptive":
                counts = np.array([v["completions"] for v in r["trace"]])
                idx = np.flatnonzero(np.diff(counts, prepend=0) > 0)
                idx = idx[y[idx] > 0]
                axes[j].plot(x[idx], y[idx], marker="o", ms=3, mfc="white", mec=COLORS[m], mew=.6, ls="none")
    target = np.linalg.norm([1.1, -.8, 1.4, -1.2, .9, -.7] +
                           [.45**r * b for r, b in zip([2, 2, 3, 3], [-1.3, 1.5, .9, -1.1])])
    axes[1].axhline(target, color=".45", ls=":", lw=.8)
    for j, (ax, title) in enumerate(zip(axes, ("(a) Accuracy", "(b) Stored coefficients"))):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(.85, 3600)
        lo, hi = min(extrema[j]), max(extrema[j])
        ax.set_ylim(max(lo * .55, 1e-16), hi * 1.5)
        ax.set_title(title)
        ax.grid(axis="y", color=".86", lw=.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    axes[0].set_ylabel(r"Relative $H^1$ error")
    axes[1].set_ylabel(r"Stored coefficient $\ell^2$ norm")
    fig.supxlabel("Coefficient fits (log scale)", x=.55, y=.04, fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=1.2, label=LABELS[m]) for m in COLORS]
    handles += [Line2D([0], [0], color=".2", marker="o", mfc="white", ls="none", ms=3.5, label="Completion"),
                Line2D([0], [0], color=".2", marker=">", ls="none", ms=3.5, label="Fit limit"),
                Line2D([0], [0], color=".2", marker="x", ls="none", ms=3.5, label=r"Return with $E>10^{-6}$")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.53, 1.01), columnspacing=1.1)
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_order.{ext}", bbox_inches="tight", pad_inches=.04, dpi=220,
                    metadata={"Title": "Guarded directional recovery: all ten matched starts"})
    plt.close(fig)
    (out / "figure_order_provenance.json").write_text(json.dumps({
        "inputs": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "historical_standard": True, "all_guarded_starts_included": True,
        "endpoint_markers": "Independently evaluated at 80 decimal digits; intermediate histories use solver diagnostics",
        "terminal_marker_rule": "Triangle for fit limit; otherwise cross if final relative H1 error exceeds 1e-6, dot if successful.",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
