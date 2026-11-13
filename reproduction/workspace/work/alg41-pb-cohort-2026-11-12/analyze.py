#!/usr/bin/env python3
"""Statistics, table rows and the main-text figure for Algorithm 4.1 on the PB cohort (records only)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "adaptive-desingularization" / "experiments"))
import plot_style as pstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ("pg_t", "pg_h", "lbfgs_t", "lbfgs_h")
LABEL = {"pg_t": r"Algorithm 4.1 ($t$)", "pg_h": r"Projected gradient in $h$",
         "lbfgs_t": r"L-BFGS-B in $t$", "lbfgs_h": r"L-BFGS-B in $h$"}
COLOR = {"pg_t": pstyle.RED, "pg_h": pstyle.GREEN, "lbfgs_t": pstyle.RED, "lbfgs_h": pstyle.GREEN}
STYLE = {"pg_t": "-", "pg_h": "-", "lbfgs_t": "--", "lbfgs_h": "--"}
TOL = 1e-7


def load():
    runs = {a: [] for a in ARMS}
    for p in sorted((HERE / "results").glob("*/*.json")):
        r = json.loads(p.read_text())
        if r.get("stop") == "not_run":          # outside the seed subsets of amendment 2
            continue
        runs[r["arm"]].append(r)
    return runs


def fits_to(trace, tau):
    for row in trace:
        if row["residual"] <= tau:
            return row["fits"]
    return None


def rate(trace, start=1e-4):
    """Least-squares slope of log10 residual against iteration on the tail where residual <= start."""
    tail = [(r["iteration"], r["residual"]) for r in trace if r["residual"] <= start and r["residual"] > 0]
    if len(tail) < 8:
        return None
    it = np.array([a for a, _ in tail], float)
    lr = np.log10([b for _, b in tail])
    if lr.max() - lr.min() < 1.0:
        return None
    A = np.vstack([it, np.ones_like(it)]).T
    coef, res, *_ = np.linalg.lstsq(A, lr, rcond=None)
    pred = A @ coef
    r2 = 1 - float(((lr - pred) ** 2).sum()) / float(((lr - lr.mean()) ** 2).sum())
    return dict(q=float(10 ** coef[0]), r2=r2, decades=float(lr.max() - lr.min()), points=len(tail))


def power_exponent(trace, k0=100):
    """Slope of log residual against log iteration for iterations >= k0 (sublinear model r ~ k^-p)."""
    pts = [(r["iteration"], r["residual"]) for r in trace if r["iteration"] >= k0 and r["residual"] > 0]
    if len(pts) < 8:
        return None
    x = np.log([a for a, _ in pts]); y = np.log([b for _, b in pts])
    p = np.polyfit(x, y, 1)
    return float(-p[0])


def summarize(runs):
    out = {}
    for a in ARMS:
        rr = [r for r in runs[a] if r.get("stop") != "exception"]
        exc = len(runs[a]) - len(rr)
        reach = [r for r in rr if r["stop"] == "tolerance"]
        row = dict(n=len(runs[a]), exceptions=exc, reached=len(reach),
                   median_fits_stop=float(np.median([r["fits"] for r in rr])) if rr else None,
                   median_fits_tol=float(np.median([r["fits"] for r in reach])) if reach else None,
                   max_fits_tol=int(max(r["fits"] for r in reach)) if reach else None,
                   median_E=float(np.median([r["final_relative_error_110"] for r in rr])) if rr else None,
                   max_E=float(max(r["final_relative_error_110"] for r in rr)) if rr else None,
                   median_res=float(np.median([r["final_residual"] for r in rr])) if rr else None,
                   zero_gap_pairs=int(sum(r["zero_gaps"] for r in rr)), pairs=4 * len(rr),
                   runs_with_zero_gap=int(sum(1 for r in rr if r["zero_gaps"] > 0)),
                   switches=int(sum(r.get("switches", 0) for r in rr)),
                   runs_with_switch=int(sum(1 for r in rr if r.get("switches", 0) > 0)),
                   separated=int(sum(1 for r in rr if r["separation_margin"] > 0 and r["box_clipped_trials"] == 0)),
                   min_margin=float(min(r["separation_margin"] for r in rr)) if rr else None,
                   stops=dict(zip(*np.unique([r["stop"] for r in rr], return_counts=True))) if rr else {})
        row["stops"] = {str(k): int(v) for k, v in row["stops"].items()}
        for tau in (1e-5, 1e-6, 1e-7):
            f = [fits_to(r["trace"], tau) for r in rr]
            got = [x for x in f if x is not None]
            row[f"reached_{tau:.0e}"] = len(got)
            row[f"median_fits_{tau:.0e}"] = float(np.median(got)) if got else None
        if a in ("pg_t",):
            rates = [rate(r["trace"]) for r in rr]
            q = [x["q"] for x in rates if x]
            r2 = [x["r2"] for x in rates if x]
            row["rate"] = dict(estimated=len(q), median_q=float(np.median(q)) if q else None,
                               q_quartiles=[float(np.percentile(q, 25)), float(np.percentile(q, 75))] if q else None,
                               min_r2=float(min(r2)) if r2 else None, median_r2=float(np.median(r2)) if r2 else None)
        if a in ("pg_h",):
            ex = [power_exponent(r["trace"]) for r in rr]
            ex = [x for x in ex if x is not None]
            row["sublinear_exponent"] = dict(median=float(np.median(ex)) if ex else None,
                                             range=[float(min(ex)), float(max(ex))] if ex else None)
        per_d = {}
        for d in (5, 10, 20):
            dd = [r for r in rr if r["d"] == d]
            per_d[d] = dict(n=len(dd), reached=sum(r["stop"] == "tolerance" for r in dd),
                            median_fits_stop=float(np.median([r["fits"] for r in dd])) if dd else None,
                            median_E=float(np.median([r["final_relative_error_110"] for r in dd])) if dd else None)
        row["per_dimension"] = per_d
        out[a] = row
    # paired comparisons on common starts
    key = lambda r: (r["d"], r["geometry"], r["seed"])
    def paired(a, b, metric):
        A = {key(r): r for r in runs[a] if r.get("stop") != "exception"}
        B = {key(r): r for r in runs[b] if r.get("stop") != "exception"}
        common = sorted(set(A) & set(B))
        xa = np.array([metric(A[k]) for k in common], float); xb = np.array([metric(B[k]) for k in common], float)
        ok = np.isfinite(xa) & np.isfinite(xb)
        res = dict(n=int(ok.sum()), a_smaller=int((xa[ok] < xb[ok]).sum()), ties=int((xa[ok] == xb[ok]).sum()))
        if ok.sum() >= 5 and np.any(xa[ok] != xb[ok]):
            res["wilcoxon_p"] = float(stats.wilcoxon(xa[ok], xb[ok]).pvalue)
            res["median_ratio"] = float(np.median(xa[ok] / xb[ok]))
        return res
    out["paired"] = dict(
        lbfgs_fits_t_vs_h=paired("lbfgs_t", "lbfgs_h", lambda r: r["fits"] if r["stop"] == "tolerance" else np.nan),
        lbfgs_E_t_vs_h=paired("lbfgs_t", "lbfgs_h", lambda r: r["final_relative_error_110"]),
        pg_residual_t_vs_h=paired("pg_t", "pg_h", lambda r: r["final_residual"]),
        pgt_vs_lbfgsh_fits=paired("pg_t", "lbfgs_h", lambda r: r["fits"] if r["stop"] == "tolerance" else np.nan),
    )
    return out


def tex_sci(x):
    if x is None:
        return "--"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"


def latex_rows(summary):
    names = {"pg_t": r"\cref{alg:reversible-pairs} ($t$)", "pg_h": r"Projected gradient in $h$",
             "lbfgs_t": r"L-BFGS-B in $t$", "lbfgs_h": r"L-BFGS-B in $h$"}
    rows = []
    for a in ("pg_t", "pg_h", "lbfgs_t", "lbfgs_h"):
        s = summary[a]
        fits = s["median_fits_tol"] if s["reached"] else s["median_fits_stop"]
        star = "" if s["reached"] else "$^*$"
        rows.append(f"{names[a]} & {s['n'] - s['exceptions']} & {s['reached']} & {fits:.0f}{star} & "
                    f"{tex_sci(s['median_E'])} & {tex_sci(s['median_res'])}\\\\")
    return "\n".join(rows)


def latex_rows_per_dimension(runs):
    names = {"pg_t": r"\cref{alg:reversible-pairs} ($t$)", "pg_h": r"Projected gradient in $h$",
             "lbfgs_t": r"L-BFGS-B in $t$", "lbfgs_h": r"L-BFGS-B in $h$"}
    rows = []
    for a in ("pg_t", "pg_h", "lbfgs_t", "lbfgs_h"):
        for d in (5, 10, 20):
            rr = [r for r in runs[a] if r["d"] == d and r.get("stop") != "exception"]
            if not rr:
                continue
            reach = [r for r in rr if r["stop"] == "tolerance"]
            fits = np.median([r["fits"] for r in (reach or rr)])
            rows.append(f"{names[a]} & {d} & {len(rr)} & {len(reach)} & {fits:.0f}{'' if reach else '$^*$'} & "
                        f"{tex_sci(float(np.median([r['final_relative_error_110'] for r in rr])))} & "
                        f"{tex_sci(float(np.median([r['final_residual'] for r in rr])))} & "
                        f"{sum(r['zero_gaps'] for r in rr)}/{4 * len(rr)} & "
                        f"{sum(1 for r in rr if r['separation_margin'] > 0 and r['box_clipped_trials'] == 0)}\\\\")
    return "\n".join(rows)


def figure(runs, path):
    pstyle.apply_style(7.5)
    pair = json.loads((ROOT / "paper_1111" / "revision-validation" / "reversible_pair_checks.json").read_text())["results"]
    closing = next(r for r in pair if r["problem"] == "positive_gap_derivative_s099")
    opening = next(r for r in pair if r["problem"] == "negative_slope_reopens_from_t0")
    fig, axes = plt.subplots(2, 2, figsize=(5.1, 3.9), layout="constrained")
    (a1, a2), (a3, a4) = axes
    for ax, rec, title in ((a1, closing, "(a) Positive slope: closing"),
                           (a2, opening, "(b) Negative slope: reopening")):
        tr = rec["trace"]
        ax.plot([r["iteration"] for r in tr], [float(r["t"]) for r in tr], color=pstyle.RED, linewidth=1.3)
        ax.set(xlabel="Accepted steps", ylabel=r"Squared half-gap $t$", title=title)
        ax.set_ylim(bottom=-0.03 * max(float(r["t"]) for r in tr))
        ax.grid(alpha=0.2, linewidth=0.5)
    k0 = closing["first_zero_iteration"]
    a1.plot(k0, 0, "o", color=pstyle.BLACK, markersize=3.5, zorder=3)
    a1.annotate(rf"$t=0$ from step {k0}", xy=(k0, 0), xytext=(k0 + 0.5, 0.017), fontsize=6.5,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5})
    for ev in opening["events"]:
        a2.plot(ev["iteration"], float(ev["t"]), "s", color=pstyle.BLACK, markersize=3.5, zorder=3,
                label="Storage change")
    a2.legend(loc="lower right", frameon=False)
    # (c) one cohort start, four arms
    rep = (10, 0, 10)
    for a in ARMS:
        r = next((x for x in runs[a] if (x["d"], x["geometry"], x["seed"]) == rep), None)
        if r is None:
            continue
        tr = r["trace"]
        a3.semilogy([x["fits"] for x in tr], [x["residual"] for x in tr], color=COLOR[a], linestyle=STYLE[a],
                    linewidth=1.2, label=LABEL[a])
    a3.axhline(TOL, color=pstyle.GRAY, linewidth=0.7, linestyle=":")
    a3.set(xlabel="Coefficient fits", ylabel=r"Canonical residual $\|\mathcal{R}_1\|_2$",
           title=r"(c) PB cohort, one start ($d=10$)")
    a3.legend(frameon=False, loc="upper right", fontsize=6)
    a3.grid(alpha=0.2, linewidth=0.5)
    # (d) all starts: Algorithm 4.1 (linear) against projected gradient in h
    for a, alpha in (("pg_h", 0.35), ("pg_t", 0.18)):
        for i, r in enumerate(runs[a]):
            if r.get("stop") == "exception":
                continue
            tr = r["trace"]
            a4.semilogy([x["iteration"] for x in tr], [x["residual"] for x in tr], color=COLOR[a],
                        linewidth=0.6, alpha=alpha, label=(LABEL[a] if i == 0 else None))
    a4.axhline(TOL, color=pstyle.GRAY, linewidth=0.7, linestyle=":")
    n_t = sum(1 for r in runs["pg_t"] if r.get("stop") != "exception")
    n_h = sum(1 for r in runs["pg_h"] if r.get("stop") != "exception")
    a4.set(xlabel="Iterations", ylabel=r"Canonical residual $\|\mathcal{R}_1\|_2$",
           title=rf"(d) All starts: {n_t} in $t$, {n_h} in $h$")
    leg = a4.legend(frameon=False, loc="upper right", fontsize=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1); lh.set_linewidth(1.2)
    a4.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(path, metadata={"Creator": "alg41-pb-cohort analyze.py; saved records only"})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    runs = load()
    summary = summarize(runs)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    for a in ARMS:
        s = summary[a]
        print(a, {k: v for k, v in s.items() if k not in ("per_dimension",)})
        print("   per d", s["per_dimension"])
    print("paired", json.dumps(summary["paired"], indent=1))
    (HERE / "table_rows.tex").write_text(latex_rows(summary) + "\n")
    (HERE / "table_rows_per_dimension.tex").write_text(latex_rows_per_dimension(runs) + "\n")
    figure(runs, HERE / "figure_alg41_cohort.pdf")
    files = sorted((HERE / "results").rglob("*.json"))
    (HERE / "inputs.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE)}\n"
                                                for p in files))


if __name__ == "__main__":
    main()
