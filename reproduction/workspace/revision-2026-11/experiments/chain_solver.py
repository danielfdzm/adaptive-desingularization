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
         They span the original atoms exactly (Newton's formula), stay bounded as the cluster collapses, and
         an atom joining the block leaves the existing columns unchanged when p_i <- p_i / (t_i - s).
Detection pairs the two closest free atoms; an atom closer to the axis of a pair, chain or line block than
across it (|s| >= |q|) joins it, closest first, if the enlarged block has diameter at most sigma.  A flagged pair or line block of order n is
completed into a chain of order n after the state and energy tests.  S never regroups; NC forms pairs
and joins atoms (stable coordinates) but never completes; C also completes.  "standard_restart" is S with the stopping rule of NC and C (restart check).
"""
from __future__ import annotations

import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_n] = "1"

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
PB = HERE.parents[1] / "adaptive-desingularization" / "final-experiments" / "pb"
sys.path.insert(0, str(PB)); sys.path.insert(0, str(HERE))
from fit import solve_coefficients                      # noqa: E402  (original Newton coefficient solver)
from quadrature import bregman_parts                    # noqa: E402
import chain_kernels as ck                              # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
GL_X, GL_W = roots_legendre(8)                           # pair basis D_h on [-1, 1]
NQ = 4                                                   # nodes per variable in line blocks (checked with 8)
_RULES = {}


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


# ----------------------------------------------------------------------------------------------------
class Geometry:
    """Packing of the nonlinear parameters; unit directions use tangent coordinates rebuilt per restart.

    The optimizer sees the parameters divided by `scale`.  L-BFGS-B starts a bounded problem with the step
    -gradient, which in the original variables is -scale^2 * gradient; Solver.restart_geometry chooses
    `scale` so that this first step has length 0.01 sigma.  Later steps use quasi-Newton curvature."""

    def __init__(self, blocks, d, sigma, box, scale=1.0):
        self.blocks, self.d, self.SCALE = blocks, d, scale
        self.frames, vals, self.bounds = [], [], []
        for b in blocks:
            vals.extend(b["c"]); self.bounds.extend([(-box, box)] * d)
            if b["kind"] == "atom":
                self.frames.append(None)
                continue
            U = null_space(b["e"].reshape(1, d))
            self.frames.append(U)
            vals.extend([0.0] * (d - 1)); self.bounds.extend([(-2.0, 2.0)] * (d - 1))
            if b["kind"] == "pair" or (b["kind"] == "line" and b["base"] == "pair"):
                vals.append(b["h"]); self.bounds.append((1e-12 * sigma, 0.6 * sigma))
            if b["kind"] == "line":
                vals.extend(b["s"]); self.bounds.extend([(-0.6 * sigma, 0.6 * sigma)] * len(b["s"]))
                for p in b["p"]:
                    vals.extend(list(U.T @ p)); self.bounds.extend([(-1e6, 1e6)] * (d - 1))
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
                if b["kind"] == "pair" or (b["kind"] == "line" and b["base"] == "pair"):
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


def primitives(blocks, d):
    """Centres, unit directions and orders of all primitives, and the weight matrix WS (primitives x columns)
    such that column j of the model equals sum_i WS[i, j] d_{u_i}^{o_i} G(. - c_i)."""
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

    for b in blocks:
        c = torch.as_tensor(b["c"])
        if b["kind"] == "atom":
            contrib(add(c, e0, 0), col, 1.0)
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
                    ti, pi_ = nodes[i], tb["p"][ii]
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
    C, Ud, O, WS = primitives(blocks, d)
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
        Mt, bt, Pt = assemble(blocks, self.d, self.sigma, self.X, self.target)
        M, b, Phi = (v.detach().numpy() for v in (Mt, bt, Pt))
        g = self.fit(M, b, Phi, self.warm)
        obj, residual = self.objective_at(M, b, Phi, g)
        gt = torch.as_tensor(g)
        proxy = (.5 * gt @ Mt @ gt - bt @ gt + torch.as_tensor(self.Wq * residual) @ (Pt @ gt)) / self.target2
        grad = torch.autograd.grad(proxy, pt)[0].numpy()
        self.cache = {"p": np.array(p), "blocks": blocks, "coef": g, "obj": obj, "grad": grad, "M": M, "b": b}
        return obj, grad

    def accept_cached(self):
        self.blocks = [to_numpy(b) for b in self.cache["blocks"]]
        self.coef = self.cache["coef"].copy(); self.warm = self.coef.copy()

    def fit_current(self):
        geo = Geometry(self.blocks, self.d, self.sigma, self.box)
        self.cache = None
        self.evaluate(geo.x, geo)
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
    def join(self, b, g, atom_c, w):
        """Line block and coefficients after an atom w G(. - atom_c) joins block b (Newton's formula)."""
        e = b["e"]
        v = atom_c - b["c"]; s = float(v @ e); q = v - s * e
        nodes = block_nodes(b)
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
        p_old = [new["p"][i] / (nodes[nu + i] - s) for i in range(len(new["s"]))]
        new["s"] = np.append(new["s"], s)
        new["p"] = np.array(p_old + [q / omega[n + 1]]).reshape(len(new["s"]), self.d)
        new["k"] = n + 1
        new["id"] = self._id()
        return new, g_new, s, q

    def regroup(self):
        changed = False
        coef = [self.coef[s].copy() for s in self.col_slices(self.blocks)]
        blocks = [dict(b) for b in self.blocks]
        used = set()
        if self.method in ("stable", "adaptive"):           # joins into pairs, chains (C only) and line blocks
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
                        if abs(prod(s - t for t in nodes)) < 1e-14 * self.sigma ** len(nodes):
                            continue
                        if best is None or dist < best[0]:
                            best = (dist, ia)
                if best is None:
                    continue
                ia = best[1]
                new, g_new, s, q = self.join(b, coef[ib], blocks[ia]["c"], float(coef[ia][0]))
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
                    if 0 < dist <= self.sigma and (best is None or dist < best[0]):
                        best = (dist, i, j)
            if best is None:
                break
            dist, i, j = best
            ci, cj, wi, wj = blocks[i]["c"], blocks[j]["c"], float(coef[i][0]), float(coef[j][0])
            h = dist / 2
            pair = {"kind": "pair", "c": (ci + cj) / 2, "e": (ci - cj) / dist, "h": h, "id": self._id()}
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
        if b["kind"] == "pair":
            pts, orders = [c + b["h"] * e, c - b["h"] * e], [0, 0]
        else:
            nodes, nu = block_nodes(b), base_size(b)
            if b["base"] == "pair":
                pts, orders = [c + b["h"] * e, c - b["h"] * e], [0, 0]
            else:
                pts, orders = [c] * nu, list(range(nu))
            for i in range(nu, len(nodes)):
                lam = prod(nodes[i] - nodes[j] for j in range(len(nodes)) if j != i)
                pts.append(c + nodes[i] * e + lam * b["p"][i - nu]); orders.append(0)
        P = np.array(pts)
        extent = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1)))
        return P, np.array(orders), extent

    def a_priori_bound(self, b, g):
        """State change of the completion that keeps the coefficients (eqs. column-bound, multidim-pair-transfer)."""
        N = self.norms
        if b["kind"] == "pair":
            h = b["h"]
            return h * h * (abs(g[0]) * N[2] / 2 + abs(g[1]) * N[3] / 6)
        nodes, nu, n = block_nodes(b), base_size(b), b["k"]
        bound = 0.0
        if b["base"] == "pair":
            h = b["h"]
            bound += h * h * (abs(g[0]) * N[2] / 2 + abs(g[1]) * N[3] / 6)
        for r in range(nu, n + 1):
            delta = max(abs(t) for t in nodes[:r + 1])
            trans = sum(abs(prod(nodes[i] - nodes[j] for j in range(r + 1, n + 1))) * float(np.linalg.norm(b["p"][i - nu]))
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
        return False

    def block_gram(self, old, new):
        """Gram matrices of the columns of two blocks (closed form, double precision)."""
        Ca, Ua, Oa, Sa = primitives([old], self.d)
        Cb, Ub, Ob, Sb = primitives([new], self.d)
        g = lambda C1, U1, O1, S1, C2, U2, O2, S2: (S1.T @ ck.h1_gram(C1, U1, O1, C2, U2, O2, self.sigma) @ S2).numpy()
        return (g(Ca, Ua, Oa, Sa, Ca, Ua, Oa, Sa), g(Cb, Ub, Ob, Sb, Ca, Ua, Oa, Sa), g(Cb, Ub, Ob, Sb, Cb, Ub, Ob, Sb))

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
        self.events.append({"kind": "completion" if ok else "rejected", "fits": self.fits, "from": b["kind"],
                            "base": b.get("base"), "order": new["k"], "omega": float(om), "a_priori_bound": float(bound),
                            "centre_shift": float(shift), "tolerance": self.tol_rel * norm,
                            "dJ_normalized": after - before, "eps_normalized": eps})
        if not ok:
            self.rejections += 1
            return False
        self.blocks = trial; self.coef = coef_trial; self.allowance_index += 1; self.completions += 1; self.flags = {}
        self.warm = self.coef.copy(); self.fit_current(); self.record()
        return True

    # ---- main loop -------------------------------------------------------------------------------
    def run(self):
        stop = "work_limit"
        try:
            self.fit_current(); self.record()
            self.scan()
            stalls = 0
            while self.fits < self.max_fits:
                geometry = self.restart_geometry()

                def callback(x):
                    self.evaluate(x, geometry); self.accept_cached()
                    self.iterations += 1
                    if self.iterations % self.scan_every == 0:
                        self.record()
                        if self.scan():
                            raise StructureChange
                try:
                    result = minimize(lambda p: self.evaluate(p, geometry), geometry.x, jac=True, method="L-BFGS-B",
                                      bounds=geometry.bounds, callback=callback,
                                      options={"ftol": 1e-15, "gtol": 1e-12, "maxls": 40, "maxiter": 6000})
                except StructureChange:
                    stalls = 0
                    continue
                self.evaluate(result.x, geometry); self.accept_cached(); self.record()
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
        return self.summary(stop)

    def state_primitives(self):
        C, U, O, WS = primitives(self.blocks, self.d)
        w = (WS @ torch.as_tensor(self.coef)).numpy()
        return [(C[i].numpy(), U[i].numpy(), int(O[i]), float(w[i])) for i in range(len(O)) if w[i] != 0.0]

    def summary(self, stop):
        A = self.state_primitives()
        B = [(c, u, o, -w) for c, u, o, w in self.target_np]
        err2 = ck.mp_norm2(A + B, self.sigma, self.d, dps=80)
        rel = float((max(err2, 0) / ck.mp_norm2(self.target_np, self.sigma, self.d, dps=80)) ** 0.5)
        return {"method": self.method, "d": self.d, "fits": self.fits, "iterations": self.iterations, "stop": stop,
                "final_relative_error": rel,
                "trace_final_error_double": self.trace[-1]["relative_error"] if self.trace else None,
                "coefficient_norm": float(np.linalg.norm(self.coef)), "completions": self.completions,
                "joins": self.joins, "rejections": self.rejections, "blocks_final": self.order_counts(),
                "wall": time.perf_counter() - self.t0, "trace": self.trace, "events": self.events,
                "final_blocks": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()} for b in self.blocks],
                "final_coefficients": self.coef.tolist()}
