#!/usr/bin/env python3
"""Sequential collinear completion for fixed-scale Gaussian models in R^d (revision, Nov 2026).

Blocks, with F(t) = G(. - c - t e) and |e| = 1; the linear coefficients are refitted by variable projection
at every trial configuration:
  atom   G(. - c)                                                                   1 slot
  pair   E_h = (F(h) + F(-h))/2, D_h = (F(h) - F(-h))/(2h)                           2 slots
  chain  Phi_r = F^(r)(0) = (-1)^r d_e^r G(. - c), r = 0..k   (completed block of order k)   k+1 slots
  line   a pair or chain (base nodes t_0..t_{nu-1}: +-h, or k+1 zeros) with joined atoms at c + t_i e + q_i,
         i = nu..n, q_i = lambda_i p_i, lambda_i = prod_{j != i} (t_i - t_j).  Columns: the base columns and
           Psi_r = r![t_0..t_r]F + r! sum_{nu <= i <= r} pi_{i,r} T_i,      r = nu..n,
           pi_{i,r} = prod_{r < j <= n} (t_i - t_j),
           T_i = (G(. - c - t_i e - q_i) - F(t_i)) / lambda_i = -E_tau (p_i . grad) G(. - c - t_i e - tau q_i).
         They span the original atoms exactly (Newton's formula); Eq. (C.4) guards all joins and trials.
         Stored p is dimensionless p_hat = sigma**(n-1) p_dimensional. The columns stay bounded under the guard, and
         an atom joining the block leaves the existing columns unchanged when p_i <- p_i / (t_i - s).
Detection pairs the two closest free atoms; an atom closer to the axis of a pair, chain or line block than
across it (|s| >= |q|) joins it, closest first, if the enlarged block has diameter at most sigma.  A flagged pair or line block of order n is
completed into a chain of order n after the state and energy tests.  S never regroups; NC forms pairs
and joins atoms (stable coordinates) but never completes; C also completes.  "standard_restart" is S with the stopping rule of NC and C (restart check).

Phase 1 revision (11 Nov 2026), method "tpair" (Algorithm 1 of the revised paper):
  * every pair is stored and optimized in the squared half-gap t = h^2 in [0, (0.6 sigma)^2]; its columns
    E_t, D_t are even in h, evaluated by the h-columns with h = sqrt(t) for t >= T_SWITCH sigma^2 and by
    their Taylor series in t (orders <= 5, truncation below 1e-17 relative) below.  A pair reaches its
    derivative limit (t = 0) or reopens through the bound constraint, without a discrete transfer;
  * joined (line) blocks of order 2-3 are completed as in "adaptive", with one more acceptance test: the
    opening slope sigma^2 dJ/dtau of the completed chain along the node-spread path must be at least
    -tol_rel ||v||_H^2 (all energies normalized by ||u*||_H^2);
  * a t-pair with h = sqrt(t) < 1e-12 sigma joins as a chain of order one (state change below 1e-24).
Methods "standard", "standard_restart", "stable" and "adaptive" are unchanged (bit-identical paths).
Amendment (11 Nov 2026, all methods): a failed coefficient solve at an optimizer trial is an inadmissible trial
(+inf, the line search backtracks; Solver.evaluate_trial); it ends the run only at an accepted state.  Runs that
never meet such a trial are unaffected.
"""
from __future__ import annotations

import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"

from collections import Counter
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import roots_jacobi, roots_legendre

HERE = Path(__file__).resolve().parent
PB = HERE
sys.path.insert(0, str(PB)); sys.path.insert(0, str(HERE))
from fit import solve_coefficients                      # noqa: E402  (original Newton coefficient solver)
from quadrature import bregman_parts                    # noqa: E402
import chain_kernels as ck                              # noqa: E402
from directional_guard import (InadmissibleChart, UNIT_ROUNDOFF, guard_offsets, guard_stored)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
GL_X, GL_W = roots_legendre(8)                           # pair basis D_h on [-1, 1]
NQ = 4                                                   # nodes per variable in line blocks (checked with 8)
_RULES = {}
T_SWITCH = 5e-6                                          # t-pairs: series in t below T_SWITCH * sigma^2
T_SERIES = 2                                             # E_t, D_t series to t^T_SERIES (orders <= 5)
H_JOIN_MIN = 1e-12                                       # t-pairs with sqrt(t) below this * sigma join as chains


def pair_halfgap(b):
    """Half-gap of a pair block: stored h (h-chart) or sqrt(t) (t-chart of method tpair)."""
    if "t" in b:
        t = b["t"]
        if torch.is_tensor(t):
            return torch.sqrt(torch.clamp(t, min=0.0))
        return math.sqrt(max(float(t), 0.0))
    return b["h"]


def beta_rule(a, b):
    """NQ-point Gauss--Jacobi rule for Beta(a, b) on [0, 1], weights summing to one."""
    key = ("beta", a, b, NQ)
    if key not in _RULES:
        x, w = roots_jacobi(NQ, b - 1, a - 1)
        _RULES[key] = ((x + 1) / 2, w / w.sum())
    return _RULES[key]


def dirichlet_rule(mult):
    """Product rule for Lambda ~ Dirichlet(mult) by stick breaking, U_k ~ Beta(mult_k, sum_{j>k} mult_j).

    By the Hermite--Genocchi formula, r![t_0..t_r]F = E F^(r)(sum_k Lambda_k tau_k) for the distinct nodes
    tau_k of t_0..t_r with multiplicities mult_k."""
    key = ("dirichlet", tuple(mult), NQ)
    if key not in _RULES:
        rules = [beta_rule(mult[k], sum(mult[k + 1:])) for k in range(len(mult) - 1)]
        pts, wts = [], []
        for combo in itertools.product(*[range(NQ)] * len(rules)):
            lam, rest, w = [], 1.0, 1.0
            for k, i in enumerate(combo):
                u = rules[k][0][i]; w *= rules[k][1][i]
                lam.append(rest * u); rest *= 1 - u
            lam.append(rest)
            pts.append(lam); wts.append(w)
        _RULES[key] = (np.array(pts), np.array(wts))
    return _RULES[key]


def gl01():
    key = ("gl01", NQ)
    if key not in _RULES:
        x, w = roots_legendre(NQ)
        _RULES[key] = ((x + 1) / 2, w / 2)
    return _RULES[key]


class StructureChange(Exception):
    pass


class WorkLimit(Exception):
    pass


def base_size(b):
    return 2 if b["base"] == "pair" else b["kb"] + 1


def ncols(b):
    kind = b["kind"]
    if kind == "atom":
        return 1
    if kind == "pair":
        return 2
    return b["k"] + 1


def line_nodes(b):
    """Axial nodes t_0..t_n of a line block, with repetition (torch scalars or floats)."""
    if b["base"] == "pair":
        nodes = [b["h"], -b["h"]]
    else:
        nodes = [0.0 * b["s"][0]] * (b["kb"] + 1)
    return nodes + [b["s"][i] for i in range(len(b["s"]))]


def block_nodes(b):
    """Axial nodes (floats) of a pair, chain or line block."""
    if b["kind"] == "pair" and "t" in b:
        h = float(pair_halfgap(b))
        return [h, -h]
    if b["kind"] == "pair":
        return [float(b["h"]), -float(b["h"])]
    if b["kind"] == "chain":
        return [0.0] * (b["k"] + 1)
    return [float(t) for t in line_nodes(b)]


def prod(xs):
    out = 1.0
    for x in xs:
        out = out * x
    return out


def validate_blocks(blocks, sigma):
    """No column/fit/gradient may be evaluated for an inadmissible chart."""
    for b in blocks:
        nb = to_numpy(b)
        for key in ("c", "e", "h", "s", "p", "t"):
            if key in nb and not np.all(np.isfinite(nb[key])):
                raise InadmissibleChart("nonfinite_block")
        if nb["kind"] == "line":
            guard_stored(block_nodes(nb), nb["p"], base_size(nb), sigma)


# ----------------------------------------------------------------------------------------------------
class Geometry:
    """Packing of the nonlinear parameters; unit directions use tangent coordinates rebuilt per restart.

    The optimizer sees the parameters divided by `scale`.  L-BFGS-B starts a bounded problem with the step
    -gradient, which in the original variables is -scale^2 * gradient; Solver.restart_geometry chooses
    `scale` so that this first step has length 0.01 sigma.  Later steps use quasi-Newton curvature."""

    def __init__(self, blocks, d, sigma, box, scale=1.0):
        self.blocks, self.d, self.SCALE = blocks, d, scale
        validate_blocks(blocks, sigma)
        self.frames, vals, self.bounds = [], [], []
        for b in blocks:
            vals.extend(b["c"]); self.bounds.extend([(-box, box)] * d)
            if b["kind"] == "atom":
                self.frames.append(None)
                continue
            U = null_space(b["e"].reshape(1, d))
            self.frames.append(U)
            vals.extend([0.0] * (d - 1)); self.bounds.extend([(-2.0, 2.0)] * (d - 1))
            if b["kind"] == "pair" and "t" in b:
                vals.append(b["t"]); self.bounds.append((0.0, (0.6 * sigma) ** 2))
            elif b["kind"] == "pair" or (b["kind"] == "line" and b["base"] == "pair"):
                vals.append(b["h"]); self.bounds.append((1e-12 * sigma, 0.6 * sigma))
            if b["kind"] == "line":
                vals.extend(b["s"]); self.bounds.extend([(-0.6 * sigma, 0.6 * sigma)] * len(b["s"]))
                for p in b["p"]:
                    vals.extend(list(U.T @ p)); self.bounds.extend([(-1.0, 1.0)] * (d - 1))
        self.x = np.array(vals, dtype=float) / self.SCALE
        self.bounds = [(lo / self.SCALE, hi / self.SCALE) for lo, hi in self.bounds]

    def unpack(self, p):
        p = p * self.SCALE
        out, j, d = [], 0, self.d
        for b, U in zip(self.blocks, self.frames):
            nb = {k: v for k, v in b.items() if k in ("kind", "id", "k", "kb", "base")}
            nb["c"] = p[j:j + d]; j += d
            if b["kind"] != "atom":
                Ut = torch.as_tensor(U)
                e = torch.as_tensor(b["e"]) + (Ut @ p[j:j + d - 1] if d > 1 else 0.0); j += d - 1
                e = e / torch.linalg.vector_norm(e)
                nb["e"] = e
                if b["kind"] == "pair" and "t" in b:
                    nb["t"] = p[j]; j += 1
                elif b["kind"] == "pair" or (b["kind"] == "line" and b["base"] == "pair"):
                    nb["h"] = p[j]; j += 1
                if b["kind"] == "line":
                    J = len(b["s"])
                    nb["s"] = p[j:j + J]; j += J
                    Up = Ut - e[:, None] * (e @ Ut)[None, :] if d > 1 else Ut
                    ps = []
                    for _ in range(J):
                        ps.append(Up @ p[j:j + d - 1] if d > 1 else torch.zeros(d)); j += d - 1
                    nb["p"] = torch.stack(ps)
            out.append(nb)
        return out


def primitives(blocks, d, sigma=1.0):
    """Centres, unit directions and orders of all primitives, and the weight matrix WS (primitives x columns)
    such that column j of the model equals sum_i WS[i, j] d_{u_i}^{o_i} G(. - c_i)."""
    validate_blocks(blocks, sigma)
    C, U, O, rows, cols, vals = [], [], [], [], [], []
    e0 = torch.zeros(d); e0[0] = 1.0
    col = 0

    def add(c, u, o):
        C.append(c); U.append(u); O.append(o)
        return len(C) - 1

    def contrib(i, j, w):
        rows.append(i); cols.append(j); vals.append(w if torch.is_tensor(w) else torch.tensor(float(w)))

    def pair_columns(c, e, h, j):
        contrib(add(c + h * e, e, 0), j, 0.5); contrib(add(c - h * e, e, 0), j, 0.5)
        for x, w in zip(GL_X, GL_W):
            contrib(add(c + float(x) * h * e, e, 1), j + 1, -0.5 * float(w))

    def tpair_columns(c, e, t, j):
        # E_t = sum_k t^k/(2k)! d_e^{2k} G,  D_t = -sum_k t^k/(2k+1)! d_e^{2k+1} G  (even in h, smooth in t)
        if float(t.detach()) >= T_SWITCH * sigma * sigma:
            pair_columns(c, e, torch.sqrt(t), j)
        else:
            for k in range(T_SERIES + 1):
                contrib(add(c, e, 2 * k), j, t ** k / math.factorial(2 * k))
                contrib(add(c, e, 2 * k + 1), j + 1, -(t ** k) / math.factorial(2 * k + 1))

    for b in blocks:
        c = torch.as_tensor(b["c"])
        if b["kind"] == "atom":
            contrib(add(c, e0, 0), col, 1.0)
        elif b["kind"] == "pair" and "t" in b:
            tpair_columns(c, torch.as_tensor(b["e"]), torch.as_tensor(b["t"]), col)
        elif b["kind"] == "pair":
            pair_columns(c, torch.as_tensor(b["e"]), torch.as_tensor(b["h"]), col)
        elif b["kind"] == "chain":
            e = torch.as_tensor(b["e"])
            for r in range(b["k"] + 1):
                contrib(add(c, e, r), col + r, (-1.0) ** r)
        else:
            e = torch.as_tensor(b["e"]); n = b["k"]; nu = base_size(b)
            tb = {kk: (torch.as_tensor(v) if kk in ("h", "s", "p") else v) for kk, v in b.items()}
            if b["base"] == "pair":
                pair_columns(c, e, tb["h"], col)
                distinct, mult = [tb["h"], -tb["h"]], [1, 1]
            else:
                for r in range(nu):
                    contrib(add(c, e, r), col + r, (-1.0) ** r)
                distinct, mult = [0.0 * tb["s"][0]], [nu]
            nodes = line_nodes(tb)
            for r in range(nu, n + 1):         # r![t_0..t_r]F = E F^(r)(sum_k Lambda_k tau_k)
                tau = distinct + [tb["s"][i] for i in range(r - nu + 1)]
                lam, wts = dirichlet_rule(mult + [1] * (r - nu + 1))
                for lrow, w in zip(lam, wts):
                    x = sum(float(lk) * tk for lk, tk in zip(lrow, tau))
                    contrib(add(c + x * e, e, r), col + r, ((-1.0) ** r) * float(w))
            if d > 1:                          # r! pi_{i,r} T_i, T_i = -E_tau (p_i . grad) G(. - c - t_i e - tau q_i)
                gx, gw = gl01()
                for ii in range(n + 1 - nu):
                    i = nu + ii
                    ti, pi_ = nodes[i], tb["p"][ii] * sigma ** (1 - n)
                    q = prod(ti - nodes[j] for j in range(n + 1) if j != i) * pi_
                    factors = {r: math.factorial(r) * prod(ti - nodes[j] for j in range(r + 1, n + 1))
                               for r in range(i, n + 1)}
                    for x, w in zip(gx, gw):
                        cen = c + ti * e + float(x) * q
                        for a in range(d):
                            ea = torch.zeros(d); ea[a] = 1.0
                            k_ = add(cen, ea, 1)
                            for r, f in factors.items():
                                contrib(k_, col + r, -float(w) * pi_[a] * f)
        col += ncols(b)
    WS = torch.zeros(len(C), col).index_put((torch.as_tensor(rows), torch.as_tensor(cols)), torch.stack(vals),
                                             accumulate=True)
    return torch.stack(C), torch.stack(U), np.array(O), WS


def assemble(blocks, d, sigma, X, target):
    C, Ud, O, WS = primitives(blocks, d, sigma)
    M = WS.T @ ck.h1_gram(C, Ud, O, C, Ud, O, sigma) @ WS
    tc, tu, to, tw = target
    b = WS.T @ (ck.h1_gram(C, Ud, O, tc, tu, to, sigma) @ tw)
    return M, b, ck.point_values(X, C, Ud, O, sigma) @ WS


def to_numpy(b):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = float(v.detach()) if v.dim() == 0 else v.detach().numpy().copy()
        out[k] = v
    return out


# ----------------------------------------------------------------------------------------------------
class Solver:
    def __init__(self, d, sigma, target, starts, method, X, Wq, lam=1.0, max_fits=3000, box=2.0,
                 scan_every=20, persistence=2, d_on=0.5, kappa_on=0.1, tol_rel=0.01, eps_scale=1e-3, max_order=3):
        self.d, self.sigma, self.method, self.max_fits, self.box = d, sigma, method, max_fits, box
        self.scan_every, self.persistence, self.d_on, self.kappa_on = scan_every, persistence, d_on, kappa_on
        self.tol_rel, self.eps_scale, self.max_order = tol_rel, eps_scale, max_order
        tc, tu, to, tw = target
        self.target = (torch.as_tensor(tc, dtype=torch.float64), torch.as_tensor(tu, dtype=torch.float64),
                       np.asarray(to), torch.as_tensor(tw, dtype=torch.float64))
        self.target_np = [(np.asarray(c, float), np.asarray(u, float), int(o), float(w)) for c, u, o, w in zip(tc, tu, to, tw)]
        self.X = torch.as_tensor(X, dtype=torch.float64)
        self.Wq = lam * np.asarray(Wq, float)
        self.Uq = (ck.point_values(self.X, *self.target[:3], sigma) @ self.target[3]).numpy()
        self.target2 = float(self.target[3] @ ck.h1_gram(*self.target[:3], *self.target[:3], sigma) @ self.target[3])
        self.norms = {r: ck.derivative_norm(r, sigma, d) for r in range(5)}
        self.next_id = 0
        self.blocks = [{"kind": "atom", "c": np.asarray(c, float), "id": self._id()} for c in starts]
        self.coef, self.warm, self.cache = None, None, None
        self.fits, self.iterations, self.completions, self.joins, self.rejections = 0, 0, 0, 0, 0
        self.allowance_index, self.flags, self.trace, self.events = 0, {}, [], []
        self.guard_counts = Counter()
        self.guard_examples = []
        self.solve_seconds = None
        self.t0 = time.perf_counter()

    def _id(self):
        self.next_id += 1
        return self.next_id

    # ---- objective --------------------------------------------------------------------------------
    def fit(self, M, b, Phi, g0):
        if self.fits >= self.max_fits:
            raise WorkLimit
        self.fits += 1
        if g0 is None or len(g0) != len(b):
            g0 = np.linalg.lstsq(M, b, rcond=None)[0]
        g, _ = solve_coefficients(M / self.target2, b / self.target2, np.ones(len(b)), None, Phi, self.Uq,
                                  self.Wq / self.target2, g0)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("nonfinite coefficient fit")
        return g

    def objective_at(self, M, b, Phi, g):
        gap, residual, _ = bregman_parts(Phi @ g, self.Uq)
        return float((.5 * g @ M @ g - b @ g + self.Wq @ gap) / self.target2), residual

    def evaluate(self, p, geometry):
        if self.cache is not None and np.array_equal(p, self.cache["p"]):
            return self.cache["obj"], self.cache["grad"]
        pt = torch.tensor(p, requires_grad=True)
        blocks = geometry.unpack(pt)
        try:
            validate_blocks(blocks, self.sigma)
        except InadmissibleChart as exc:
            self.guard_rejection("optimizer_trial", exc)
            return float("inf"), np.zeros_like(p)
        Mt, bt, Pt = assemble(blocks, self.d, self.sigma, self.X, self.target)
        M, b, Phi = (v.detach().numpy() for v in (Mt, bt, Pt))
        g = self.fit(M, b, Phi, self.warm)
        obj, residual = self.objective_at(M, b, Phi, g)
        gt = torch.as_tensor(g)
        proxy = (.5 * gt @ Mt @ gt - bt @ gt + torch.as_tensor(self.Wq * residual) @ (Pt @ gt)) / self.target2
        grad = torch.autograd.grad(proxy, pt)[0].numpy()
        self.cache = {"p": np.array(p), "blocks": blocks, "coef": g, "obj": obj, "grad": grad, "M": M, "b": b}
        return obj, grad

    def evaluate_trial(self, p, geometry):
        """Objective at an L-BFGS-B trial (Phase 1, amendment of 11 Nov 2026, all methods).

        A singular or otherwise failed coefficient solve at a trial point makes the trial inadmissible: the trial
        returns +inf and the line search backtracks, as for a trial rejected by the chart guard.  Accepted states are
        evaluated by evaluate(), where the same failure still ends the run.  Runs without such a trial follow exactly
        the same path as before the amendment."""
        try:
            return self.evaluate(p, geometry)
        except FloatingPointError as exc:
            self.guard_counts[f"optimizer_trial:{exc}"] += 1
            if len(self.guard_examples) < 30:
                self.guard_examples.append({"context": "optimizer_trial", "reason": str(exc), "fits": self.fits})
            return float("inf"), np.zeros_like(p)

    def guard_rejection(self, context, exc):
        self.guard_counts[f"{context}:{exc.reason}"] += 1
        if len(self.guard_examples) < 30:
            self.guard_examples.append({"context": context, "reason": exc.reason, "fits": self.fits})

    def accept_cached(self):
        self.blocks = [to_numpy(b) for b in self.cache["blocks"]]
        self.coef = self.cache["coef"].copy(); self.warm = self.coef.copy()

    def fit_current(self):
        geo = Geometry(self.blocks, self.d, self.sigma, self.box)
        self.cache = None
        obj, _ = self.evaluate(geo.x, geo)
        if not np.isfinite(obj):
            raise InadmissibleChart("invalid_restart")
        self.accept_cached()

    def restart_geometry(self, first_step=0.01):
        """Geometry for a new L-BFGS-B call whose first step -scale^2 * gradient has length first_step * sigma."""
        self.warm = self.coef.copy()
        unit = Geometry(self.blocks, self.d, self.sigma, self.box)
        self.cache = None
        _, grad = self.evaluate(unit.x, unit)
        gnorm = float(np.linalg.norm(grad))
        scale = float(np.clip(math.sqrt(first_step * self.sigma / max(gnorm, 1e-300)), 1e-3, 1e3))
        geometry = Geometry(self.blocks, self.d, self.sigma, self.box, scale)
        self.cache["p"], self.cache["grad"] = geometry.x.copy(), grad * scale
        return geometry

    def rel_error(self):
        M, b, g = self.cache["M"], self.cache["b"], self.coef
        return math.sqrt(max(g @ M @ g - 2 * b @ g + self.target2, 0.0) / self.target2)

    def order_counts(self):
        out = {}
        for b in self.blocks:
            key = b["kind"] if b["kind"] in ("atom", "pair") else f'{b["kind"]}{b["k"]}'
            out[key] = out.get(key, 0) + 1
        return out

    def record(self):
        row = {"fits": self.fits, "iterations": self.iterations, "wall": time.perf_counter() - self.t0,
               "relative_error": self.rel_error(), "coefficient_norm": float(np.linalg.norm(self.coef)),
               "completions": self.completions, "joins": self.joins, "blocks": self.order_counts()}
        self.trace.append(row)
        return row

    def col_slices(self, blocks):
        out, j = [], 0
        for b in blocks:
            out.append(slice(j, j + ncols(b))); j += ncols(b)
        return out

    # ---- exact changes of coordinates ------------------------------------------------------------
    def join_base(self, b):
        """Join-equivalent form of a t-pair: an h-chart pair with h = sqrt(t), or, when sqrt(t) < H_JOIN_MIN sigma,
        the chain of order one with the same coefficients (E_0 = G, D_0 = -d_e G; state change O(t) < 1e-24)."""
        if b["kind"] != "pair" or "t" not in b:
            return b
        h = float(pair_halfgap(b))
        if h < H_JOIN_MIN * self.sigma:
            return {"kind": "chain", "c": b["c"], "e": b["e"], "k": 1, "id": b["id"]}
        return {"kind": "pair", "c": b["c"], "e": b["e"], "h": h, "id": b["id"]}

    def join(self, b, g, atom_c, w):
        """Line block and coefficients after an atom w G(. - atom_c) joins block b (Newton's formula)."""
        if b["kind"] == "pair" and "t" in b:
            t0 = float(b["t"])
            b = self.join_base(b)
            self.events.append({"kind": "tpair_join", "fits": self.fits, "t": t0, "as": b["kind"]})
        e = b["e"]
        v = atom_c - b["c"]; s = float(v @ e); q = v - s * e
        nodes = block_nodes(b)
        products, offsets = self.join_eligibility(b, atom_c)
        if b["kind"] == "pair":
            new = {"kind": "line", "base": "pair", "c": b["c"].copy(), "e": e.copy(), "h": float(b["h"]),
                   "s": np.zeros(0), "p": np.zeros((0, self.d))}
        elif b["kind"] == "chain":
            new = {"kind": "line", "base": "chain", "kb": b["k"], "c": b["c"].copy(), "e": e.copy(),
                   "s": np.zeros(0), "p": np.zeros((0, self.d))}
        else:
            new = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in b.items()}
        n = len(nodes) - 1
        omega = [prod(s - t for t in nodes[:r]) for r in range(n + 2)]          # omega_r(s), r = 0..n+1
        g_new = np.array(g, dtype=float).copy()
        first = 0
        if new["base"] == "pair":                                            # base columns E_h, D_h
            g_new[0] += w; g_new[1] += w * s
            first = 2
        for r in range(first, n + 1):
            g_new[r] += w * omega[r] / math.factorial(r)
        g_new = np.append(g_new, w * omega[n + 1] / math.factorial(n + 1))
        nu = base_size(new)
        new["s"] = np.append(new["s"], s)
        # All enlarged-list products and physical offsets passed (C.4) before this quotient.
        new["p"] = (np.asarray(offsets) / self.sigma) / products[:, None]
        new["k"] = n + 1
        validate_blocks([new], self.sigma)
        new["id"] = self._id()
        return new, g_new, s, q

    def join_eligibility(self, b, atom_c):
        b = self.join_base(b)
        validate_blocks([b], self.sigma)
        nodes = block_nodes(b)
        v = np.asarray(atom_c) - b["c"]
        s = float(v @ b["e"]); q = v - s * b["e"]
        if not np.all(np.isfinite(v)):
            raise InadmissibleChart("nonfinite_candidate")
        if b["kind"] == "line":
            nu = base_size(b)
            old_products, _ = guard_stored(nodes, b["p"], nu, self.sigma)
            offsets = list(self.sigma * old_products[:, None] * b["p"])
        else:
            nu = len(nodes); offsets = []
        offsets.append(q)
        products, _ = guard_offsets(nodes + [s], offsets, nu, self.sigma)
        return products, offsets

    def regroup(self):
        changed = False
        coef = [self.coef[s].copy() for s in self.col_slices(self.blocks)]
        blocks = [dict(b) for b in self.blocks]
        used = set()
        if self.method in ("stable", "adaptive", "tpair"):  # joins into pairs, chains (C only) and line blocks
            for ib, b in enumerate(blocks):
                if ib in used or b["kind"] == "atom" or (b["kind"] != "pair" and b["k"] >= self.max_order):
                    continue
                nodes = block_nodes(b)
                P = self.block_functions(b)[0] if b["kind"] != "chain" else b["c"][None]
                best = None
                for ia, a in enumerate(blocks):
                    if a["kind"] != "atom" or ia in used:
                        continue
                    v = a["c"] - b["c"]; s = float(v @ b["e"]); q = v - s * b["e"]
                    dist = float(np.linalg.norm(v))
                    diam = max(float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1))),
                               float(np.max(np.linalg.norm(P - a["c"], axis=-1))))
                    if diam <= self.sigma and abs(s) >= np.linalg.norm(q):
                        try:
                            self.join_eligibility(b, a["c"])
                        except InadmissibleChart as exc:
                            self.guard_rejection("join_candidate", exc)
                            continue
                        if best is None or dist < best[0]:
                            best = (dist, ia)
                if best is None:
                    continue
                ia = best[1]
                try:
                    new, g_new, s, q = self.join(b, coef[ib], blocks[ia]["c"], float(coef[ia][0]))
                except InadmissibleChart as exc:
                    self.guard_rejection("join_transfer", exc)
                    continue
                blocks[ib] = new; coef[ib] = g_new; used.update({ib, ia})
                self.joins += 1; changed = True
                self.events.append({"kind": "join", "fits": self.fits, "into": b["kind"], "base": new["base"],
                                    "order": new["k"], "s": s, "q": float(np.linalg.norm(q))})
        keep = [i for i in range(len(blocks)) if not (i in used and blocks[i]["kind"] == "atom")]
        blocks = [blocks[i] for i in keep]; coef = [coef[i] for i in keep]
        while True:                                            # pairs: closest free atoms, one at a time
            dir_c = [b["c"] for b in blocks if b["kind"] != "atom"]
            free = [i for i, b in enumerate(blocks) if b["kind"] == "atom"
                    and all(np.linalg.norm(b["c"] - c) > self.sigma for c in dir_c)]
            best = None
            for x in range(len(free)):
                for z in range(x + 1, len(free)):
                    i, j = free[x], free[z]
                    dist = float(np.linalg.norm(blocks[i]["c"] - blocks[j]["c"]))
                    if dist <= 32 * UNIT_ROUNDOFF * self.sigma:
                        self.guard_rejection("pair_candidate", InadmissibleChart("unresolved_pair_displacement"))
                        continue
                    if np.isfinite(dist) and dist <= self.sigma and (best is None or dist < best[0]):
                        best = (dist, i, j)
            if best is None:
                break
            dist, i, j = best
            ci, cj, wi, wj = blocks[i]["c"], blocks[j]["c"], float(coef[i][0]), float(coef[j][0])
            h = dist / 2
            pair = {"kind": "pair", "c": (ci + cj) / 2, "e": (ci - cj) / dist, "h": h, "id": self._id()}
            if self.method == "tpair":                     # same pair, stored by its squared half-gap
                pair = {"kind": "pair", "c": pair["c"], "e": pair["e"], "t": h * h, "id": pair["id"]}
            blocks = [b for t, b in enumerate(blocks) if t not in (i, j)] + [pair]
            coef = [cf for t, cf in enumerate(coef) if t not in (i, j)] + [np.array([wi + wj, h * (wi - wj)])]
            changed = True
            self.events.append({"kind": "pair", "fits": self.fits, "h": h})
        if changed:
            self.blocks, self.coef = blocks, np.concatenate(coef); self.warm = self.coef.copy()
        return changed

    # ---- detection and completion ----------------------------------------------------------------
    def kappa(self, centres, dirs, orders):
        H = ck.h1_gram(centres, dirs, orders, centres, dirs, orders, self.sigma).numpy()
        s = np.sqrt(np.diag(H))
        return math.sqrt(max(float(np.linalg.eigvalsh(H / s[:, None] / s[None, :]).min()), 0.0))

    def trial_energy(self, blocks, g):
        M, b, Phi = (v.detach().numpy() for v in assemble(blocks, self.d, self.sigma, self.X, self.target))
        return self.objective_at(M, b, Phi, g)[0], M

    def block_functions(self, b):
        """Atoms or derivatives of a pair or line block (centres, orders) and the block diameter."""
        c, e = b["c"], b["e"]
        if b["kind"] == "pair" and "t" in b:
            h = float(pair_halfgap(b))
            pts, orders = [c + h * e, c - h * e], [0, 0]
        elif b["kind"] == "pair":
            pts, orders = [c + b["h"] * e, c - b["h"] * e], [0, 0]
        else:
            nodes, nu = block_nodes(b), base_size(b)
            if b["base"] == "pair":
                pts, orders = [c + b["h"] * e, c - b["h"] * e], [0, 0]
            else:
                pts, orders = [c] * nu, list(range(nu))
            for i in range(nu, len(nodes)):
                lam = prod(nodes[i] - nodes[j] for j in range(len(nodes)) if j != i)
                pts.append(c + nodes[i] * e + lam * b["p"][i - nu] * self.sigma ** (1 - b["k"])); orders.append(0)
        P = np.array(pts)
        extent = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1)))
        return P, np.array(orders), extent

    def a_priori_bound(self, b, g):
        """State change of the completion that keeps the coefficients (eqs. column-bound, multidim-pair-transfer)."""
        N = self.norms
        if b["kind"] == "pair":
            h = float(pair_halfgap(b)) if "t" in b else b["h"]
            return h * h * (abs(g[0]) * N[2] / 2 + abs(g[1]) * N[3] / 6)
        nodes, nu, n = block_nodes(b), base_size(b), b["k"]
        bound = 0.0
        if b["base"] == "pair":
            h = b["h"]
            bound += h * h * (abs(g[0]) * N[2] / 2 + abs(g[1]) * N[3] / 6)
        for r in range(nu, n + 1):
            delta = max(abs(t) for t in nodes[:r + 1])
            trans = sum(abs(prod(nodes[i] - nodes[j] for j in range(r + 1, n + 1))) * float(np.linalg.norm(b["p"][i - nu])) * self.sigma ** (1 - n)
                        for i in range(nu, r + 1)) if self.d > 1 else 0.0
            bound += abs(g[r]) * (delta * N[r + 1] + math.factorial(r) * trans * N[1])
        return bound

    def scan(self):
        if self.method in ("standard", "standard_restart"):
            return False
        if self.regroup():
            self.flags = {}
            self.fit_current(); self.record()
            return True
        sl = self.col_slices(self.blocks)
        for ib, b in enumerate(self.blocks):
            if b["kind"] not in ("pair", "line"):
                continue
            P, orders, extent = self.block_functions(b)
            dirs = torch.as_tensor(np.repeat(b["e"][None], len(orders), 0))
            flag = extent <= self.d_on * self.sigma and self.kappa(torch.as_tensor(P), dirs, orders) <= self.kappa_on
            self.flags[b["id"]] = self.flags.get(b["id"], 0) + 1 if flag else 0
            if self.method == "adaptive" and self.flags[b["id"]] >= self.persistence:
                if self.complete(ib, sl):
                    return True
            # tpair: pairs reach t = 0 continuously; only joined blocks are completed discretely
            if self.method == "tpair" and b["kind"] == "line" and self.flags[b["id"]] >= self.persistence:
                if self.complete(ib, sl):
                    return True
        return False

    def block_gram(self, old, new):
        """Gram matrices of the columns of two blocks (closed form, double precision)."""
        Ca, Ua, Oa, Sa = primitives([old], self.d, self.sigma)
        Cb, Ub, Ob, Sb = primitives([new], self.d, self.sigma)
        g = lambda C1, U1, O1, S1, C2, U2, O2, S2: (S1.T @ ck.h1_gram(C1, U1, O1, C2, U2, O2, self.sigma) @ S2).numpy()
        return (g(Ca, Ua, Oa, Sa, Ca, Ua, Oa, Sa), g(Cb, Ub, Ob, Sb, Ca, Ua, Oa, Sa), g(Cb, Ub, Ob, Sb, Cb, Ub, Ob, Sb))

    def opening_slope(self, blocks, ib, g0):
        """Normalized opening slope of the completed chain blocks[ib] (order n >= 1), after one coefficient refit.

        Spreading the n+1 nodes of a chain of order n symmetrically about its centre, with Gaussian weights that
        keep its first n+1 Taylor coefficients, changes the state by tau * w1 + O(tau^2), where tau is half the sum
        of the squared node offsets and
            w1 = c_{n-1} / (n (n+1)) d_e^{n+1} G + c_n / ((n+1)(n+2)) d_e^{n+2} G,
        with c_m the coefficient of d_e^m G in the chain (for a pair: (a/2) d_e^2 G - (c/6) d_e^3 G).
        Returns sigma^2 dJ/dtau and ||v||_H^2, both divided by ||u*||_H^2.  The refit counts as one fit."""
        M, b, Phi = (v.detach().numpy() for v in assemble(blocks, self.d, self.sigma, self.X, self.target))
        g = self.fit(M, b, Phi, g0)
        chain = blocks[ib]
        n = chain["k"]
        gc = g[self.col_slices(blocks)[ib]]
        cm = [(-1.0) ** r * float(gc[r]) for r in range(n + 1)]       # chain columns are (-1)^r d_e^r G
        c = torch.as_tensor(chain["c"])[None]
        e = torch.as_tensor(chain["e"])[None]
        Cw, Uw, Ow = torch.cat([c, c]), torch.cat([e, e]), np.array([n + 1, n + 2])
        ww = torch.tensor([cm[n - 1] / (n * (n + 1)), cm[n] / ((n + 1) * (n + 2))])
        C, U, O, WS = primitives(blocks, self.d, self.sigma)
        vw = WS @ torch.as_tensor(g)
        tc, tu, to, tw = self.target
        lin = float(vw @ ck.h1_gram(C, U, O, Cw, Uw, Ow, self.sigma) @ ww
                    - tw @ ck.h1_gram(tc, tu, to, Cw, Uw, Ow, self.sigma) @ ww)
        _, residual, _ = bregman_parts(Phi @ g, self.Uq)
        w1q = (ck.point_values(self.X, Cw, Uw, Ow, self.sigma) @ ww).numpy()
        nonlin = float(self.Wq @ (residual * w1q))
        return self.sigma ** 2 * (lin + nonlin) / self.target2, float(g @ M @ g) / self.target2

    def stationarity_diagnostics(self):
        """Projected-gradient max-abs (unit scale, w.r.t. the L-BFGS-B bounds; coefficients refitted) and the
        normalized coefficient KKT residual at the final state.  Computed after the run, outside the fit count."""
        saved = (self.fits, self.max_fits, self.cache, self.warm)
        out = {}
        try:
            self.max_fits = float("inf")
            geo = Geometry(self.blocks, self.d, self.sigma, self.box)
            self.cache, self.warm = None, self.coef.copy()
            obj, grad = self.evaluate(geo.x, geo)
            lo = np.array([a for a, _ in geo.bounds]); hi = np.array([c for _, c in geo.bounds])
            x = geo.x
            active = ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0))
            pg = np.where(active, 0.0, grad)
            M, b, Phi = (v.detach().numpy() for v in assemble(self.blocks, self.d, self.sigma, self.X, self.target))
            _, residual, _ = bregman_parts(Phi @ self.coef, self.Uq)
            gcoef = M @ self.coef - b + Phi.T @ (self.Wq * residual)
            kkt = float(np.max(np.abs(gcoef) / np.sqrt(np.diag(M)))) / math.sqrt(self.target2)
            out = {"projected_gradient_max_abs": float(np.max(np.abs(pg))) if len(pg) else 0.0,
                   "gradient_max_abs": float(np.max(np.abs(grad))) if len(grad) else 0.0,
                   "active_bounds": int(np.sum((x <= lo) | (x >= hi))),
                   "normalized_coefficient_kkt": kkt, "final_objective_normalized": float(obj)}
            slopes = []                                   # opening slopes of every completed chain at the end
            for ib, b in enumerate(self.blocks):
                if b["kind"] == "chain":
                    slope, v2 = self.opening_slope(self.blocks, ib, self.coef)
                    slopes.append({"order": int(b["k"]), "slope_normalized": slope, "threshold": -self.tol_rel * v2})
            out["final_chain_slopes"] = slopes
        except Exception as exc:                          # diagnostics must never change the run outcome
            out = {"stationarity_diagnostics_error": repr(exc)}
        finally:
            self.fits, self.max_fits, self.cache, self.warm = saved
        return out

    def complete(self, ib, sl):
        """Completion of a flagged pair or line block on the same slots.

        Pairs: midpoint, coefficients kept, omega = a priori bound (eq. multidim-pair-transfer).
        Line blocks of order n: candidate centres are the block centre and the centroid of the axial nodes; the
        trial is the H-orthogonal projection of the block onto span{d_e^r G(. - centre): r <= n}, and omega is
        its computed state change (closed-form Gram matrices), never larger than the recorded a priori bound."""
        b, g = self.blocks[ib], self.coef[sl[ib]]
        bound = self.a_priori_bound(b, g)
        if b["kind"] == "pair":
            om, shift, g_new = bound, 0.0, g.copy()
            new = {"kind": "chain", "c": b["c"].copy(), "e": b["e"].copy(), "k": 1}
        else:
            nodes = block_nodes(b)
            best = None
            for x in (0.0, sum(nodes) / len(nodes)):
                cand = {"kind": "chain", "c": b["c"] + x * b["e"], "e": b["e"].copy(), "k": b["k"]}
                A, Bx, Cg = self.block_gram(b, cand)
                gp = np.linalg.solve(Cg, Bx @ g)
                om = math.sqrt(max(float(g @ A @ g - 2 * gp @ Bx @ g + gp @ Cg @ gp), 0.0))
                if best is None or om < best[0]:
                    best = (om, cand, x, gp)
            om, new, shift, g_new = best
        new["id"] = self._id()
        trial = self.blocks[:ib] + [new] + self.blocks[ib + 1:]
        coef_trial = np.concatenate([self.coef[:sl[ib].start], g_new, self.coef[sl[ib].stop:]])
        before, M = self.trial_energy(self.blocks, self.coef)
        after, _ = self.trial_energy(trial, coef_trial)
        norm = math.sqrt(max(self.coef @ M @ self.coef, 1e-300))
        eps = self.eps_scale * 2.0 ** (-self.allowance_index - 1)
        ok = om <= self.tol_rel * norm and (after - before) <= eps
        slope_info = {}
        if ok and self.method == "tpair":               # opening-slope test (Phase 1); old methods unchanged
            slope, vnorm2 = self.opening_slope(trial, ib, coef_trial)
            threshold = -self.tol_rel * vnorm2
            ok = slope >= threshold
            slope_info = {"slope_normalized": slope, "slope_threshold": threshold, "slope_ok": bool(ok)}
        self.events.append({"kind": "completion" if ok else "rejected", "fits": self.fits, "from": b["kind"],
                            "base": b.get("base"), "order": new["k"], "omega": float(om), "a_priori_bound": float(bound),
                            "centre_shift": float(shift), "tolerance": self.tol_rel * norm,
                            "dJ_normalized": after - before, "eps_normalized": eps, **slope_info})
        if not ok:
            self.rejections += 1
            return False
        self.blocks = trial; self.coef = coef_trial; self.allowance_index += 1; self.completions += 1; self.flags = {}
        self.warm = self.coef.copy(); self.fit_current(); self.record()
        return True

    # ---- main loop -------------------------------------------------------------------------------
    def run(self):
        self.t0 = time.perf_counter()
        stop = "work_limit"
        try:
            self.fit_current(); self.record()
            self.scan()
            stalls = 0
            while self.fits < self.max_fits:
                geometry = self.restart_geometry()

                def callback(x):
                    obj, _ = self.evaluate(x, geometry)
                    if not np.isfinite(obj):
                        return
                    self.accept_cached()
                    self.iterations += 1
                    if self.iterations % self.scan_every == 0:
                        self.record()
                        if self.scan():
                            raise StructureChange
                try:
                    result = minimize(lambda p: self.evaluate_trial(p, geometry), geometry.x, jac=True, method="L-BFGS-B",
                                      bounds=geometry.bounds, callback=callback,
                                      options={"ftol": 1e-15, "gtol": 1e-12, "maxls": 40, "maxiter": 6000})
                except StructureChange:
                    stalls = 0
                    continue
                obj, _ = self.evaluate(result.x, geometry)
                if np.isfinite(obj):
                    self.accept_cached()
                else:
                    # Keep the last accepted feasible state, never an invalid optimizer return.
                    self.fit_current()
                self.record()
                if self.method == "standard":
                    stop = "abnormal" if result.status == 2 else "function_tolerance"
                    break
                if self.scan():
                    stalls = 0
                    continue
                stalls += 1
                if stalls >= 2:
                    stop = "abnormal" if result.status == 2 else "function_tolerance"
                    break
        except WorkLimit:
            stop = "work_limit"
        except FloatingPointError as exc:
            stop = str(exc)
        self.solve_seconds = time.perf_counter() - self.t0
        return self.summary(stop)

    def state_primitives(self):
        C, U, O, WS = primitives(self.blocks, self.d, self.sigma)
        w = (WS @ torch.as_tensor(self.coef)).numpy()
        return [(C[i].numpy(), U[i].numpy(), int(O[i]), float(w[i])) for i in range(len(O)) if w[i] != 0.0]

    def summary(self, stop):
        A = self.state_primitives()
        B = [(c, u, o, -w) for c, u, o, w in self.target_np]
        err2 = ck.mp_norm2(A + B, self.sigma, self.d, dps=80)
        rel = float((max(err2, 0) / ck.mp_norm2(self.target_np, self.sigma, self.d, dps=80)) ** 0.5)
        tvals = [float(b["t"]) for b in self.blocks if b["kind"] == "pair" and "t" in b]
        extra = {"stationarity": self.stationarity_diagnostics(), "final_squared_halfgaps": tvals,
                 "zero_gap_pairs": sum(t == 0.0 for t in tvals)}
        return {**extra, "method": self.method, "d": self.d, "fits": self.fits, "iterations": self.iterations, "stop": stop,
                "final_relative_error": rel,
                "trace_final_error_double": self.trace[-1]["relative_error"] if self.trace else None,
                "coefficient_norm": float(np.linalg.norm(self.coef)), "completions": self.completions,
                "joins": self.joins, "rejections": self.rejections, "blocks_final": self.order_counts(),
                "wall": time.perf_counter() - self.t0, "solve_seconds": self.solve_seconds, "optimizer_seconds": self.solve_seconds,
                "posthoc_validation_seconds": time.perf_counter() - self.t0 - self.solve_seconds,
                "postprocessing_seconds": time.perf_counter() - self.t0 - self.solve_seconds,
                "guard_counts": dict(self.guard_counts), "guard_examples": self.guard_examples,
                "coordinate_convention": "dimensionless p_hat = (q/sigma)/prod((t_i-t_j)/sigma)",
                "trace": self.trace, "events": self.events,
                "final_blocks": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()} for b in self.blocks],
                "final_coefficients": self.coef.tolist()}
