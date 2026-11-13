#!/usr/bin/env python3
"""Varying the Gaussian budget P for prescribed (non-manufactured) forcing in one dimension.

PDE:  -u'' + u + lambda (sinh u - u) = f  on R,  model scale sigma = 1.
Energy used by every optimizer (no exact solution, no reference values):
    J(v) = 1/2 ||v||_{H^1}^2 + lambda Q[cosh v - 1 - v^2/2] - Q[f v],
with the exact whole-line Gaussian H^1 Gram matrix and a fixed 768-node Gauss--Legendre rule Q on [-12, 12],
normalized by Q[f^2].  Methods S (ordinary Gaussians, two-return rule), NC (stable coordinates) and C (pairs in
t = h^2 and completion of joined blocks) are those of the paper's chain solver (Phase-1 source, unchanged); this
file only replaces its manufactured-target objective by the forcing energy.  The reference solution, computed by
an independent boundary-value solve, is used only after optimization to measure errors.

Baselines (same energy, same reference): fixed-centre Gaussian Galerkin with P equispaced centres, and P1 finite
elements; both choose their computational interval by the smallest energy (no reference information).
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import json
import math
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_bvp
from scipy.special import roots_legendre
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "numerical-section-revision" / "phase1-2026-11-11" / "src"
sys.path.insert(0, str(SRC))
import chain_solver as cs          # noqa: E402
import chain_kernels as ck         # noqa: E402
from quadrature import bregman_parts  # noqa: E402
from fit import solve_coefficients    # noqa: E402

SIGMA = 1.0
CASES = {
    # odd single-source forcing of the paper's prescribed-forcing study (midpoint of its parameter range)
    "single_source": dict(kind="single_source", a=0.4, A=25.0, lam=10.0),
    # asymmetric two-bump sech^2 forcing (not a Gaussian combination; no symmetry)
    "sech_dipole": dict(kind="sech_dipole", a=0.5, A=12.0, x1=0.3, x2=-0.7, lam=5.0),
}
METHODS = {"S": "standard_restart", "NC": "stable", "C": "tpair"}


def forcing(case, x):
    c = CASES[case]
    if c["kind"] == "single_source":
        return c["A"] * x / c["a"] ** 2 * np.exp(-x * x / (2 * c["a"] ** 2))
    if c["kind"] == "sech_dipole":
        return c["A"] * (1 / np.cosh((x - c["x1"]) / c["a"]) ** 2 - 0.5 / np.cosh((x - c["x2"]) / c["a"]) ** 2)
    raise ValueError(case)


@lru_cache(None)
def gl(n, L=12.0):
    x, w = roots_legendre(n)
    return L * x, L * w


def remainder(v):
    """cosh v - 1 - v^2/2 without cancellation."""
    out = np.empty_like(v)
    small = np.abs(v) < 0.05
    z = v[small]
    out[small] = z ** 4 / 24 + z ** 6 / 720 + z ** 8 / 40320 + z ** 10 / 3628800
    out[~small] = np.cosh(v[~small]) - 1 - 0.5 * v[~small] ** 2
    return out


# ------------------------------------------------------------------------- reference
@lru_cache(None)
def reference(case, L=12.0, tol=1e-10):
    lam = CASES[case]["lam"]
    x = np.linspace(-L, L, 801)

    def ode(x, y):
        return np.vstack((y[1], y[0] + lam * (np.sinh(y[0]) - y[0]) - forcing(case, x)))

    def bc(ya, yb):          # exact decay conditions of the linearized operator at the truncation points
        return np.array([ya[1] - ya[0], yb[1] + yb[0]])

    guess = np.zeros((2, x.size))
    for t in (1e-6, tol):     # staged tolerance (robust for the nonlinear cases)
        r = solve_bvp(ode, bc, x, guess, tol=t, max_nodes=200000)
        if not r.success:
            raise RuntimeError(f"reference {case}: {r.message}")
        x, guess = r.x, r.y
    return r


def reference_fields(case, x):
    r = reference(case, 14.0, 3e-11)
    u, ux = r.sol(x)
    return u, ux


def reference_check(case):
    x, w = gl(1536)
    u1, ux1 = reference(case, 12.0, 1e-10).sol(x)
    u2, ux2 = reference(case, 14.0, 3e-11).sol(x)
    norm = float(w @ (u2 * u2 + ux2 * ux2))
    diff = math.sqrt(float(w @ ((u1 - u2) ** 2 + (ux1 - ux2) ** 2)) / norm)
    lam = CASES[case]["lam"]
    quad = 0.5 * norm
    nonlin = float(lam * w @ remainder(u2))
    return dict(reference_refinement_relative_H1=diff, reference_H1_norm=math.sqrt(norm),
                reference_peak=float(np.max(np.abs(u2))), nonlinear_to_quadratic_energy=nonlin / quad,
                reference_energy_normalized=float(w @ (0.5 * (u2 * u2 + ux2 * ux2) + lam * remainder(u2)
                                                       - forcing(case, x) * u2)) / float(gl(768)[1] @ forcing(case, gl(768)[0]) ** 2))


def h1_error(case, v, vx):
    x, w = gl(1536)
    u, ux = reference_fields(case, x)
    return math.sqrt(float(w @ ((v - u) ** 2 + (vx - ux) ** 2)) / float(w @ (u * u + ux * ux)))


# ------------------------------------------------------------------- chain solver
class ForcingSolver(cs.Solver):
    def __init__(self, case, P, seed, method, max_fits=3000, nodes=768, box=6.0):
        self.case, self.P, self.seed = case, P, seed
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-3.0, 3.0, P)
        X, W = gl(nodes)
        dummy = (np.zeros((1, 1)), np.ones((1, 1)), np.zeros(1, dtype=int), np.zeros(1))
        lam = CASES[case]["lam"]
        super().__init__(1, SIGMA, dummy, [np.array([s]) for s in starts], METHODS[method], X[:, None], W,
                         lam=lam, max_fits=max_fits, box=box)
        self.initial_centers = starts.tolist()
        self.base_weights = np.asarray(W, float)
        self.f = forcing(case, X)
        self.fW = torch.as_tensor(self.base_weights * self.f)
        self.target2 = float(self.base_weights @ self.f ** 2)       # normalization by Q[f^2], never a solution norm
        self.Uq = np.zeros_like(X)

    def assemble_forcing(self, blocks):
        C, U, O, WS = cs.primitives(blocks, 1, self.sigma)
        M = WS.T @ ck.h1_gram(C, U, O, C, U, O, self.sigma) @ WS
        Phi = ck.point_values(self.X, C, U, O, self.sigma) @ WS
        b = Phi.T @ self.fW
        return M, b, Phi

    def evaluate(self, p, geometry):
        if self.cache is not None and np.array_equal(p, self.cache["p"]):
            return self.cache["obj"], self.cache["grad"]
        pt = torch.tensor(p, requires_grad=True)
        blocks = geometry.unpack(pt)
        try:
            cs.validate_blocks(blocks, self.sigma)
        except cs.InadmissibleChart as exc:
            self.guard_rejection("optimizer_trial", exc)
            return float("inf"), np.zeros_like(p)
        Mt, bt, Pt = self.assemble_forcing(blocks)
        M, b, Phi = (v.detach().numpy() for v in (Mt, bt, Pt))
        g = self.fit(M, b, Phi, self.warm)
        obj, residual = self.objective_at(M, b, Phi, g)
        gt = torch.as_tensor(g)
        proxy = (0.5 * gt @ Mt @ gt - bt @ gt + torch.as_tensor(self.Wq * residual) @ (Pt @ gt)) / self.target2
        grad = torch.autograd.grad(proxy, pt)[0].numpy()
        self.cache = {"p": np.array(p), "blocks": blocks, "coef": g, "obj": obj, "grad": grad, "M": M, "b": b}
        return obj, grad

    def trial_energy(self, blocks, g):
        M, b, Phi = (v.detach().numpy() for v in self.assemble_forcing(blocks))
        return self.objective_at(M, b, Phi, g)[0], M

    def opening_slope(self, blocks, ib, g0):
        """sigma^2 dJ/dtau of a completed chain, with the forcing energy (target term -> load term)."""
        M, b, Phi = (v.detach().numpy() for v in self.assemble_forcing(blocks))
        g = self.fit(M, b, Phi, g0)
        chain = blocks[ib]
        n = chain["k"]
        gc = g[self.col_slices(blocks)[ib]]
        cm = [(-1.0) ** r * float(gc[r]) for r in range(n + 1)]
        c = torch.as_tensor(chain["c"])[None]
        e = torch.as_tensor(chain["e"])[None]
        Cw, Uw, Ow = torch.cat([c, c]), torch.cat([e, e]), np.array([n + 1, n + 2])
        ww = torch.tensor([cm[n - 1] / (n * (n + 1)), cm[n] / ((n + 1) * (n + 2))])
        C, U, O, WS = cs.primitives(blocks, 1, self.sigma)
        vw = WS @ torch.as_tensor(g)
        w1q = ck.point_values(self.X, Cw, Uw, Ow, self.sigma) @ ww
        lin = float(vw @ ck.h1_gram(C, U, O, Cw, Uw, Ow, self.sigma) @ ww - self.fW @ w1q)
        _, residual, _ = bregman_parts(Phi @ g, self.Uq)
        nonlin = float(self.Wq @ (residual * w1q.numpy()))
        return self.sigma ** 2 * (lin + nonlin) / self.target2, float(g @ M @ g) / self.target2

    def stationarity_diagnostics(self):
        saved = (self.fits, self.max_fits, self.cache, self.warm)
        out = {}
        try:
            self.max_fits = float("inf")
            geo = cs.Geometry(self.blocks, self.d, self.sigma, self.box)
            self.cache, self.warm = None, self.coef.copy()
            obj, grad = self.evaluate(geo.x, geo)
            lo = np.array([a for a, _ in geo.bounds]); hi = np.array([c for _, c in geo.bounds])
            x = geo.x
            active = ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0))
            pg = np.where(active, 0.0, grad)
            out = {"projected_gradient_max_abs": float(np.max(np.abs(pg))) if len(pg) else 0.0,
                   "active_bounds": int(np.sum((x <= lo) | (x >= hi))), "final_objective_normalized": float(obj)}
            slopes = []
            for ib, b in enumerate(self.blocks):
                if b["kind"] == "chain":
                    slope, v2 = self.opening_slope(self.blocks, ib, self.coef)
                    slopes.append({"order": int(b["k"]), "slope_normalized": slope, "threshold": -self.tol_rel * v2})
            out["final_chain_slopes"] = slopes
        except Exception as exc:
            out = {"stationarity_diagnostics_error": repr(exc)}
        finally:
            self.fits, self.max_fits, self.cache, self.warm = saved
        return out

    def rel_error(self):
        return float("nan")

    def record(self):
        row = {"fits": self.fits, "iterations": self.iterations, "wall": time.perf_counter() - self.t0,
               "energy": self.cache["obj"] if self.cache else None,
               "coefficient_norm": float(np.linalg.norm(self.coef)), "completions": self.completions,
               "joins": self.joins, "blocks": self.order_counts()}
        self.trace.append(row)

    def fields(self, x):
        C, U, O, S = cs.primitives(self.blocks, 1, self.sigma)
        X = torch.as_tensor(x[:, None])
        Pv = ck.point_values(X, C, U, O, self.sigma) @ S
        Qv = (ck.point_values(X, C, U, O + 1, self.sigma) * U[:, 0]) @ S
        c = torch.as_tensor(self.coef)
        return (Pv @ c).numpy(), (Qv @ c).numpy()

    def ordinary_weights(self):
        """Weights of the represented Gaussians (finite pairs and atoms); None if a block is completed."""
        out = []
        for b, sl in zip(self.blocks, self.col_slices(self.blocks)):
            g = self.coef[sl]
            if b["kind"] == "atom":
                out.append(g[0])
            elif b["kind"] == "pair":
                h = float(cs.pair_halfgap(b))
                if h == 0.0:
                    return None
                out.extend([g[0] / 2 + g[1] / (2 * h), g[0] / 2 - g[1] / (2 * h)])
            else:
                return None
        return np.array(out)

    def summary(self, stop):
        tvals = [float(b["t"]) for b in self.blocks if b["kind"] == "pair" and "t" in b]
        hvals = [float(b["h"]) for b in self.blocks if b["kind"] == "pair" and "h" in b]
        centres = []
        for b in self.blocks:
            if b["kind"] == "atom":
                centres.append(float(b["c"][0]))
        cen = np.sort(np.array(centres)) if centres else np.array([])
        ow = self.ordinary_weights()
        return {"method": self.method, "case": self.case, "P": self.P, "seed": self.seed, "stop": stop,
                "fits": self.fits, "iterations": self.iterations, "completions": self.completions,
                "joins": self.joins, "rejections": self.rejections, "blocks_final": self.order_counts(),
                "final_squared_halfgaps": tvals, "zero_gap_pairs": sum(t == 0.0 for t in tvals),
                "final_halfgaps": hvals, "min_atom_spacing": (float(np.min(np.diff(cen))) if cen.size > 1 else None),
                "coefficient_norm": float(np.linalg.norm(self.coef)),
                "ordinary_weight_max": (float(np.max(np.abs(ow))) if ow is not None else None),
                "final_energy_normalized": float(self.cache["obj"]) if self.cache else None,
                "stationarity": self.stationarity_diagnostics(),
                "solve_seconds": time.perf_counter() - self.t0, "initial_centres": self.initial_centers,
                "trace": self.trace, "events": self.events, "guard_counts": dict(self.guard_counts),
                "final_blocks": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
                                 for b in self.blocks],
                "final_coefficients": self.coef.tolist()}


def collision_flags(rec):
    """Collision indicators at the end of a run (see the protocol)."""
    m = rec["method"]
    if m == "tpair":
        chains = sum(v for k, v in rec["blocks_final"].items() if k.startswith("chain"))
        return dict(collided=bool(rec["zero_gap_pairs"] > 0 or rec["completions"] > 0 or chains > 0),
                    zero_gap_pairs=rec["zero_gap_pairs"], completions=rec["completions"])
    if m == "stable":
        small = sum(h <= 1e-6 * SIGMA for h in rec["final_halfgaps"])
        big = rec["ordinary_weight_max"] is None or rec["ordinary_weight_max"] > 1e6
        return dict(collided=bool(small > 0 or (rec["ordinary_weight_max"] is not None and rec["ordinary_weight_max"] > 1e6)),
                    floor_pairs=int(small))
    sp = rec["min_atom_spacing"]
    return dict(collided=bool((sp is not None and sp <= 1e-3 * SIGMA) or
                              (rec["ordinary_weight_max"] is not None and rec["ordinary_weight_max"] > 1e6)))


def run_one(case, P, seed, method, max_fits=3000):
    s = ForcingSolver(case, P, seed, method, max_fits=max_fits)
    try:
        rec = s.run()
    except Exception as exc:  # retained
        rec = dict(method=METHODS[method], case=case, P=P, seed=seed, stop="exception", exception=repr(exc))
        return rec
    x, w = gl(1536)
    v, vx = s.fields(x)
    rec["relative_H1_error"] = h1_error(case, v, vx)
    rec["energy_1536"] = float(w @ (0.5 * (v * v + vx * vx) + CASES[case]["lam"] * remainder(v) - forcing(case, x) * v)) / s.target2
    rec["collision"] = collision_flags(rec)
    rec["label"] = method
    return rec


# ------------------------------------------------------------------------ baselines
def gaussian_galerkin(case, P, Ls=(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6)):
    """P fixed equispaced Gaussians on [-L, L]; L chosen by the smallest normalized energy."""
    X, W = gl(768)
    f = forcing(case, X)
    lam = CASES[case]["lam"]
    target2 = float(W @ f * f) if False else float(W @ (f * f))
    best = None
    for L in Ls:
        c = np.linspace(-L, L, P) if P > 1 else np.array([0.0])
        C = torch.as_tensor(c[:, None]); U = torch.ones(P, 1); O = np.zeros(P, dtype=int)
        M = ck.h1_gram(C, U, O, C, U, O, SIGMA).numpy()
        Phi = ck.point_values(torch.as_tensor(X[:, None]), C, U, O, SIGMA).numpy()
        b = Phi.T @ (W * f)
        try:
            g0 = np.linalg.lstsq(M, b, rcond=None)[0]
            g, _ = solve_coefficients(M / target2, b / target2, np.ones(P), None, Phi, np.zeros_like(X),
                                      lam * W / target2, g0)
        except Exception:
            continue
        gap, _, _ = bregman_parts(Phi @ g, np.zeros_like(X))
        J = float((0.5 * g @ M @ g - b @ g + lam * W @ gap) / target2)
        if best is None or J < best[0]:
            best = (J, L, c, g)
    J, L, c, g = best
    x, w = gl(1536)
    C = torch.as_tensor(c[:, None]); U = torch.ones(P, 1); O = np.zeros(P, dtype=int)
    Xt = torch.as_tensor(x[:, None])
    v = (ck.point_values(Xt, C, U, O, SIGMA).numpy()) @ g
    vx = (ck.point_values(Xt, C, U, O + 1, SIGMA).numpy() * 1.0) @ g
    return dict(P=P, L=L, energy_normalized=J, relative_H1_error=h1_error(case, v, vx))


def fem_p1(case, n, Ls=(3, 4, 5, 6, 8, 10, 12)):
    """P1 finite elements with n interior nodes on [-L, L], zero Dirichlet data; Newton for the PB energy;
    L chosen by the smallest energy (no reference information)."""
    lam = CASES[case]["lam"]
    gq, gw = roots_legendre(4)
    Xn, Wn = gl(768)
    target2 = float(Wn @ forcing(case, Xn) ** 2)
    best = None
    for L in Ls:
        nodes = np.linspace(-L, L, n + 2)
        hh = nodes[1] - nodes[0]
        # element quadrature points and P1 shape values
        xa = nodes[:-1, None] + (gq[None, :] + 1) / 2 * hh
        wa = np.broadcast_to(gw[None, :] * hh / 2, xa.shape)
        phiL, phiR = (1 - (gq + 1) / 2), (gq + 1) / 2
        fq = forcing(case, xa)
        # stiffness + mass (exact for P1)
        main = np.full(n, 2 / hh + 4 * hh / 6)
        off = np.full(n - 1, -1 / hh + hh / 6)
        K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        # load
        load = np.zeros(n + 2)
        np.add.at(load, np.arange(n + 1), (fq * phiL[None, :] * wa).sum(1))
        np.add.at(load, np.arange(1, n + 2), (fq * phiR[None, :] * wa).sum(1))
        load = load[1:-1]
        u = np.linalg.solve(K, load)

        def fields(uin):
            full = np.concatenate([[0.0], uin, [0.0]])
            return full[:-1, None] * phiL[None, :] + full[1:, None] * phiR[None, :]

        def energy(uin):
            uq = fields(uin)
            return 0.5 * uin @ K @ uin + lam * float((remainder(uq.ravel()) * wa.ravel()).sum()) - load @ uin

        for _ in range(60):
            uq = fields(u)
            r_el = lam * (np.sinh(uq) - uq) * wa
            rhs = np.zeros(n + 2)
            np.add.at(rhs, np.arange(n + 1), (r_el * phiL[None, :]).sum(1))
            np.add.at(rhs, np.arange(1, n + 2), (r_el * phiR[None, :]).sum(1))
            grad = K @ u - load + rhs[1:-1]
            c_el = lam * (np.cosh(uq) - 1) * wa
            Hd = np.zeros(n + 2); Ho = np.zeros(n + 1)
            np.add.at(Hd, np.arange(n + 1), (c_el * phiL[None, :] ** 2).sum(1))
            np.add.at(Hd, np.arange(1, n + 2), (c_el * phiR[None, :] ** 2).sum(1))
            Ho += (c_el * phiL[None, :] * phiR[None, :]).sum(1)
            H = K + np.diag(Hd[1:-1]) + np.diag(Ho[1:-1], 1) + np.diag(Ho[1:-1], -1)
            step = np.linalg.solve(H, grad)
            E0 = energy(u); a = 1.0
            while a > 1e-10 and energy(u - a * step) > E0 - 1e-4 * a * (grad @ step):
                a *= 0.5
            u = u - a * step
            if np.linalg.norm(grad, np.inf) < 1e-12:
                break
        J = energy(u) / target2
        if best is None or J < best[0]:
            best = (J, L, nodes, u)
    J, L, nodes, u = best
    x, w = gl(1536)
    full = np.concatenate([[0.0], u, [0.0]])
    v = np.interp(x, nodes, full, left=0.0, right=0.0)
    idx = np.clip(np.searchsorted(nodes, x) - 1, 0, n)
    slope = (full[1:] - full[:-1]) / (nodes[1] - nodes[0])
    vx = np.where((x > -L) & (x < L), slope[idx], 0.0)
    return dict(dof=n, L=L, energy_normalized=J, relative_H1_error=h1_error(case, v, vx))


def source_hashes():
    paths = [Path(__file__), SRC / "chain_solver.py", SRC / "chain_kernels.py", SRC / "fit.py", SRC / "quadrature.py",
             SRC / "directional_guard.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, platform=platform.platform())
