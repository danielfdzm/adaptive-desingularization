#!/usr/bin/env python3
"""Redraw the Algorithm 4.1 comparison from saved, matched-start records.

No optimization is rerun. Nonattainment remains in the common denominator;
the work cap is never imputed as an attainment time.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import figure_style as style
style.apply_style(8.0)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
ROOT = style.ROOT
SOURCE = ROOT / "work/alg41-pb-cohort-2026-11-12/results"
ARMS = ("pg_t", "pg_h")
COLORS = {"pg_t": style.RED, "pg_h": style.BLUE}
LABELS = {
    "pg_t": r"Algorithm 4.1 ($t$)",
    "pg_h": r"Projected gradient ($h$)",
}
LINESTYLES = {"pg_t": "-", "pg_h": (0, (4, 1.8))}
TOL = 1e-7
CAP = 2500
REPRESENTATIVE = (10, 0, 10)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    runs = {arm: {} for arm in ARMS}
    paths = {arm: {} for arm in ARMS}
    for path in sorted(SOURCE.glob("*/*.json")):
        row = json.loads(path.read_text())
        if row.get("stop") == "not_run":
            continue
        arm = row["arm"]
        if arm not in ARMS:
            continue
        key = (row["d"], row["geometry"], row["seed"])
        assert key not in runs[arm], (arm, key)
        runs[arm][key] = row
        paths[arm][key] = path
    common = sorted(set.intersection(*(set(runs[arm]) for arm in ARMS)))
    assert len(common) == 18
    assert REPRESENTATIVE in common
    return runs, paths, common


def first_attainment(row):
    hits = [int(point["fits"]) for point in row.get("trace", [])
            if np.isfinite(point["residual"]) and point["residual"] <= TOL
            and point["fits"] <= CAP]
    return min(hits) if hits else None


def layout_check(fig, axes):
    """Check rendered text against data rectangles and other rendered text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        labels.append((artist.get_text(), box))
    issues = []
    for text, box in labels:
        for i, ax in enumerate(axes):
            if box.overlaps(ax.get_window_extent(renderer)):
                issues.append(f"Text {text!r} intersects data panel {i + 1}")
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            issues.append(f"Text {text!r} falls outside figure bounds")
    for i, (text, box) in enumerate(labels):
        for other, other_box in labels[i + 1:]:
            if box.overlaps(other_box):
                issues.append(f"Text overlap: {text!r}, {other!r}")
    assert not issues, "\n".join(issues)
    return {"visible_text_artists": len(labels), "text_data_intersections": 0,
            "text_text_intersections": 0, "clipped_text": 0}


def main():
    runs, paths, common = load()
    summary = {
        "description": "Matched-start comparison of the two projected-gradient methods using saved canonical residual traces",
        "tolerance": TOL, "work_cap": CAP, "common_n": len(common),
        "key_order": ["d", "geometry", "seed"], "common_keys": common,
        "representative_key": REPRESENTATIVE, "palette": COLORS,
        "attainment_rule": "First saved residual <= tolerance at fits <= cap; nonattainment stays in the denominator and is not an event",
        "arms": {}, "input_files": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 2.9))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.17, top=0.82, wspace=0.39)
    ax, bx = axes
    for panel in axes:
        panel.set_box_aspect(1)
        panel.set_xscale("log")
        panel.xaxis.set_minor_locator(NullLocator())
        panel.set_xlabel("Coefficient fits", labelpad=5)
        panel.grid(axis="y", which="major", zorder=0)
        panel.tick_params(axis="both", which="major", length=3, pad=3)
    ax.set_yscale("log")
    ax.set_xlim(0.9, 3000)
    ax.set_ylim(3e-8, 0.25)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_locator(FixedLocator([1e-7, 1e-5, 1e-3, 1e-1]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Canonical residual", labelpad=5)
    ax.set_title("(a) One matched start", loc="left", fontsize=8.3)
    ax.axhline(TOL, color=style.BLACK, linestyle=(0, (1, 2.5)), linewidth=0.8, zorder=1)
    bx.set_xlim(40, 2900)
    bx.set_ylim(-0.035, 1.05)
    bx.xaxis.set_major_locator(FixedLocator([50, 200, 1000, 2500]))
    bx.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    bx.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    bx.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    bx.set_ylabel("Fraction reaching tolerance", labelpad=5)
    bx.set_title("(b) 18 matched starts", loc="left", fontsize=8.3)
    for arm in ARMS:
        representative = runs[arm][REPRESENTATIVE]
        trace = [p for p in representative["trace"]
                 if p["fits"] > 0 and p["residual"] > 0]
        x = [p["fits"] for p in trace]
        y = [p["residual"] for p in trace]
        ax.plot(x, y, color=COLORS[arm], linestyle=LINESTYLES[arm],
                linewidth=1.2, zorder=3)
        ax.plot(x[-1], y[-1], marker="o", markersize=3, linestyle="none",
                color=COLORS[arm], zorder=4)
        hits = sorted(hit for key in common
                      if (hit := first_attainment(runs[arm][key])) is not None)
        times, multiplicities = np.unique(hits, return_counts=True)
        fraction = np.cumsum(multiplicities) / len(common)
        # Extend the empirical attainment curve only to the actual work cap.
        sx = np.r_[40, times, CAP]
        sy = np.r_[0, fraction, len(hits) / len(common)]
        bx.step(sx, sy, where="post", color=COLORS[arm],
                linestyle=LINESTYLES[arm], linewidth=1.4, zorder=3)
        bx.plot(CAP, len(hits) / len(common), marker="o", markersize=3,
                linestyle="none", color=COLORS[arm], zorder=4)
        summary["arms"][arm] = {
            "n": len(common), "attained": len(hits),
            "not_attained": len(common) - len(hits), "attainment_fits": hits,
            "fraction_at_cap": len(hits) / len(common),
            "median_fits_among_attained": float(np.median(hits)) if hits else None,
            "stops": dict(Counter(runs[arm][key]["stop"] for key in common)),
            "representative_final_fits": int(x[-1]),
            "representative_final_residual": float(y[-1]),
        }
        for key in common:
            path = paths[arm][key]
            summary["input_files"][str(path.relative_to(ROOT))] = sha256(path)
    handles = [Line2D([0], [0], color=COLORS[a], linestyle=LINESTYLES[a],
                      linewidth=1.4, label=LABELS[a]) for a in ARMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.995),
               ncol=2, columnspacing=2.3, handlelength=2.4,
               handletextpad=0.7, labelspacing=0.65, borderaxespad=0)
    summary["layout_check"] = layout_check(fig, axes)
    summary["panel_geometry"] = []
    for panel in axes:
        box = panel.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert np.isclose(box.width, box.height, rtol=0, atol=1e-12)
        summary["panel_geometry"].append({
            "width_inches": box.width, "height_inches": box.height,
            "width_to_height_ratio": box.width / box.height,
        })
    stem = HERE / "figure_alg41_cohort"
    style.save(fig, stem)
    plt.close(fig)
    shutil.copy2(stem.with_suffix(".pdf"), ROOT / "paper_1111/figure_alg41_cohort.pdf")
    summary["generator_sha256"] = sha256(Path(__file__))
    summary["style_sha256"] = sha256(HERE / "figure_style.py")
    summary["outputs_sha256"] = {ext: sha256(stem.with_suffix("." + ext))
                                  for ext in ("pdf", "svg", "png")}
    (HERE / "figure_alg41_cohort.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"common_n": len(common), "arms": summary["arms"],
                      "layout_check": summary["layout_check"]}, indent=2))


if __name__ == "__main__":
    main()
