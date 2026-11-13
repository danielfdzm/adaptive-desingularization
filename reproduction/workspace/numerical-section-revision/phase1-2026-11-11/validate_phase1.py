#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026) validation of the t-pair chain solver.  Run with the directional environment
(Python 3.13.9, NumPy 2.4.2), from any directory:

    python3.13 validate_phase1.py [--skip-regression]

Checks
  A  regression: methods "stable" and "adaptive" of the modified solver reproduce the frozen guarded records
     (second1d seed 40) exactly: errors, fits, iterations, events, traces (without wall times), final state;
  B  t-pair columns: series branch (t < T_SWITCH sigma^2) and square-root branch both match the exact pair
     functions E_h, D_h (50-digit mpmath) pointwise; the reduced-energy t-derivative at t = 0 matches a
     one-sided finite difference;
  C  opening slope: the formula of Solver.opening_slope matches finite-difference openings of chains of
     order 1, 2, 3 (nonlinear energy, lambda = 1) with moment-matched Gaussians at Hermite nodes;
  D  saddle of the paper (u* = G(.-1) + G(.+1), sigma = 1, P = 2): slope of the completed symmetric pair is
     about -1.31 per unit h^2 (unnormalized), and a tpair run leaves t = 0 and lowers the error;
  E  slope test on a joined block (sigma = 1): opening slopes of the completion of three Gaussians at -a, 0, a
     into G, G', G'' at 0 (a = 0.5, 1, 1.5; rejected only for a = 1.5 under the fixed threshold
     -tol_rel ||v||^2) and of a block whose limit is the target u* = G'' (slope 0, accepted).
Writes validation.json next to this script.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import mpmath as mp
from scipy.special import roots_hermitenorm

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))
import chain_solver as cs                     # noqa: E402
import chain_kernels as ck                    # noqa: E402
import chain_experiments as ce                # noqa: E402

FROZEN = HERE.parent / "guarded-rerun-2026-11-09" / "directional" / "accuracy-second1d"
TIME_KEYS = {"wall", "solve_seconds", "optimizer_seconds", "posthoc_validation_seconds", "postprocessing_seconds",
             "end_to_end_seconds", "setup_seconds", "timing_mode", "concurrent_workers", "timing_scope"}
NEW_KEYS = {"stationarity", "final_squared_halfgaps", "zero_gap_pairs"}


def strip_time(obj):
    if isinstance(obj, dict):
        return {k: strip_time(v) for k, v in obj.items() if k not in TIME_KEYS}
    if isinstance(obj, list):
        return [strip_time(v) for v in obj]
    return obj


def check_regression():
    out = {}
    for method in ("stable", "adaptive"):
        frozen = json.loads((FROZEN / f"second1d_d1_g0_seed40_{method}.json").read_text())
        new = json.loads(json.dumps(ce.run_one(("second1d", 1, 0, 40, method)), default=float))
        a = strip_time({k: v for k, v in frozen.items() if k not in NEW_KEYS})
        b = strip_time({k: v for k, v in new.items() if k not in NEW_KEYS})
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        out[method] = {"identical": not diff, "differing_fields": diff,
                       "final_relative_error": [frozen["final_relative_error"], new["final_relative_error"]],
                       "fits": [frozen["fits"], new["fits"]]}
    return out


def exact_pair(x, c, h, sigma):
    """E_h, D_h at points x (1D), 50 digits."""
    with mp.workdps(50):
        E, D = [], []
        for xi in x:
            g1 = mp.e ** (-(mp.mpf(xi) - c - h) ** 2 / (2 * sigma ** 2))
            g2 = mp.e ** (-(mp.mpf(xi) - c + h) ** 2 / (2 * sigma ** 2))
            E.append((g1 + g2) / 2); D.append((g1 - g2) / (2 * h))
        return np.array([float(v) for v in E]), np.array([float(v) for v in D])


def check_columns():
    sigma, c = 0.65, 0.1
    x = np.linspace(-3, 3, 61)
    X = torch.as_tensor(x[:, None])
    rows = []
    for t in (0.5 * cs.T_SWITCH * sigma ** 2, 0.999 * cs.T_SWITCH * sigma ** 2, 1.001 * cs.T_SWITCH * sigma ** 2,
              1e-3 * sigma ** 2, 0.1 * sigma ** 2):
        b = {"kind": "pair", "c": np.array([c]), "e": np.array([1.0]), "t": t, "id": 1}
        C, U, O, WS = cs.primitives([b], 1, sigma)
        vals = (ck.point_values(X, C, U, O, sigma) @ WS).numpy()
        E, D = exact_pair(x, c, math.sqrt(t), sigma)
        rows.append({"t_over_sigma2": t / sigma ** 2,
                     "max_abs_error_E": float(np.max(np.abs(vals[:, 0] - E))),
                     "max_abs_error_D": float(np.max(np.abs(vals[:, 1] - D))),
                     "branch": "series" if t < cs.T_SWITCH * sigma ** 2 else "sqrt"})
    # one-sided derivative of the reduced energy at t = 0: u* = -G' (sigma = 1, lambda = 0), pair at 0
    sigma = 1.0
    target = (np.array([[0.0]]), np.array([[1.0]]), np.array([1]), np.array([-1.0]))
    X1, W1 = np.zeros((1, 1)), np.zeros(1)
    s = cs.Solver(1, sigma, target, [np.array([-0.2]), np.array([0.2])], "tpair", X1, W1, lam=0.0, max_fits=10 ** 6)
    s.fit_current(); s.regroup(); s.fit_current()
    s.blocks[0]["t"] = 0.0; s.blocks[0]["c"] = np.array([0.05])  # off-centre, so the slope is nonzero
    geo = cs.Geometry(s.blocks, 1, sigma, s.box)
    s.cache, s.warm = None, None
    f0, g0 = s.evaluate(geo.x, geo)
    it = len(geo.x) - 1                                            # t is the last geometry coordinate
    fd = []
    for delta in (1e-6, 1e-7):
        p = geo.x.copy(); p[it] += delta
        s.cache = None
        f1, _ = s.evaluate(p, geo)
        fd.append((f1 - f0) / delta)
    return {"columns": rows, "reduced_slope_autograd": float(g0[it]), "reduced_slope_fd": fd}


def moment_open(cm, n, s, c, e):
    """Gaussians at c + s z_k e (z = zeros of He_{n+1}) whose first n+1 Taylor coefficients equal cm."""
    z, _ = roots_hermitenorm(n + 1)
    A = np.array([[(-s * zk) ** m / math.factorial(m) for zk in z] for m in range(n + 1)])
    w = np.linalg.solve(A, np.array(cm[: n + 1]))
    atoms = [{"kind": "atom", "c": np.asarray(c, float) + s * zk * np.asarray(e, float), "id": 100 + k}
             for k, zk in enumerate(z)]
    return atoms, w, 0.5 * s * s * float(np.sum(z * z))


def check_slope_formula():
    sigma, d = 0.65, 1
    target = (np.array([[0.2], [-0.9], [0.4]]), np.ones((3, 1)), np.array([2, 0, 1]), np.array([0.3, 0.7, -0.5]))
    x, w = np.polynomial.legendre.leggauss(384)
    X, W = 10.0 * x[:, None], 10.0 * w
    rows = []
    for n in (1, 2, 3):
        s = cs.Solver(d, sigma, target, [np.array([-0.8])], "tpair", X, W, lam=1.0, max_fits=10 ** 6)
        chain = {"kind": "chain", "c": np.array([0.25]), "e": np.array([1.0]), "k": n, "id": 7}
        blocks = [{"kind": "atom", "c": np.array([-0.8]), "id": 1}, chain]
        g0 = np.zeros(1 + n + 1)
        slope, vnorm2 = s.opening_slope(blocks, 1, g0)
        M, b, Phi = (v.detach().numpy() for v in cs.assemble(blocks, d, sigma, s.X, s.target))
        g = s.fit(M, b, Phi, g0)                                   # same refit as opening_slope
        base, _ = s.trial_energy(blocks, g)
        cm = [(-1.0) ** r * g[1 + r] for r in range(n + 1)]
        fds = []
        for step in (4e-2, 2e-2):
            atoms, wk, tau = moment_open(cm, n, step * sigma, [0.25], [1.0])
            opened = [blocks[0]] + atoms
            val, _ = s.trial_energy(opened, np.concatenate([g[:1], wk]))
            fds.append(sigma ** 2 * (val - base) / tau)
        richardson = (4 * fds[1] - fds[0]) / 3
        rows.append({"order": n, "formula": slope, "finite_difference": fds, "richardson": richardson,
                     "relative_difference": abs(richardson - slope) / max(abs(slope), 1e-300)})
    return rows


def check_saddle():
    sigma = 1.0
    target = (np.array([[1.0], [-1.0]]), np.ones((2, 1)), np.array([0, 0]), np.array([1.0, 1.0]))
    X1, W1 = np.zeros((1, 1)), np.zeros(1)
    s = cs.Solver(1, sigma, target, [np.array([-0.04]), np.array([0.04])], "tpair", X1, W1, lam=0.0, max_fits=10 ** 6)
    chain = {"kind": "chain", "c": np.array([0.0]), "e": np.array([1.0]), "k": 1, "id": 3}
    slope, vnorm2 = s.opening_slope([chain], 0, np.zeros(2))
    unnormalized = slope * s.target2 / sigma ** 2
    run = cs.Solver(1, sigma, target, [np.array([-0.04]), np.array([0.04])], "tpair", X1, W1, lam=0.0, max_fits=600)
    res = run.run()
    first_pair = [e for e in res["events"] if e["kind"] == "pair"]
    return {"slope_normalized": slope, "slope_per_unit_h2": unnormalized, "paper_value": -1.31,
            "tpair_run": {"final_relative_error": res["final_relative_error"], "final_t": res["final_squared_halfgaps"],
                          "initial_t": first_pair[0]["h"] ** 2 if first_pair else None, "stop": res["stop"],
                          "fits": res["fits"]}}


def check_slope_test():
    sigma = 1.0
    X1, W1 = np.zeros((1, 1)), np.zeros(1)
    out = {}
    cases = {f"three_gaussians_a{a}": (np.array([[-a], [0.0], [a]]), np.ones((3, 1)), np.array([0, 0, 0]), np.ones(3))
             for a in (0.5, 1.0, 1.5)}
    cases["second_derivative"] = (np.array([[0.0]]), np.ones((1, 1)), np.array([2]), np.array([0.5]))
    for name, target in cases.items():
        s = cs.Solver(1, sigma, target, [np.array([0.0])], "tpair", X1, W1, lam=0.0, max_fits=10 ** 6)
        chain = {"kind": "chain", "c": np.array([0.0]), "e": np.array([1.0]), "k": 2, "id": 5}
        slope, vnorm2 = s.opening_slope([chain], 0, np.zeros(3))
        out[name] = {"slope_normalized": slope, "threshold": -s.tol_rel * vnorm2,
                     "accepted": bool(slope >= -s.tol_rel * vnorm2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()
    report = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}
    if not args.skip_regression:
        report["A_regression"] = check_regression()
    report["B_columns"] = check_columns()
    report["C_slope_formula"] = check_slope_formula()
    report["D_saddle"] = check_saddle()
    report["E_slope_test"] = check_slope_test()
    (HERE / "validation.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
