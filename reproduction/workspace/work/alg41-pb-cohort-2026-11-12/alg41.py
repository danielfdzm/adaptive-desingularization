#!/usr/bin/env python3
"""Algorithm 4.1 on the first-order Poisson--Boltzmann cohort, with three comparators.

Problem data are those of the paper's first-order PB cohort (d = 5, 10, 20;
geometries 0, 1; seeds 10-34): the unchanged production modules `config`,
`quadrature`, `fit` and `kernels` give the target, the target-informed starts,
the Sobol mixture quadrature, the damped-Newton coefficient fits and the
110-digit endpoint error.  Pair columns and Gram entries use the exact parity
formulas of `squared_gap.py`, which are entire functions of the squared
half-gap t, so the same functions serve both charts (t = h^2).

The labeled blocks are the four pairs and two singletons in which the start
is generated (`config.start_centers` lists c + h e, c - h e for each pair).

Arms (all with exact-formula columns, the same coefficient solver, the same
objective and the same canonical stopping residual R_1 in (beta, t, e, c)):

  pg_t      Algorithm 4.1: full projected gradient steps in t with
            backtracking and the reversible storage rule (stable / ordinary).
  pg_h      the same projected-gradient iteration in the half-gap h.
  lbfgs_h   L-BFGS-B in h (bounds 0 <= h <= 0.6 sigma), restarted after
            abnormal returns.
  lbfgs_t   L-BFGS-B in t (bounds 0 <= t <= (0.6 sigma)^2), restarted alike.

The objective is the energy with the manufactured forcing, with the
target-only constant omitted, divided by ||u*||_H^2 (as in the production
solver).  It is solver-realizable: it uses only inner products with the
forcing, not pointwise values of u*.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.optimize import minimize
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SQDIR = ROOT / "numerical-section-revision" / "sinum-revision-2026-11-10" / "squared-gap"
SRC = SQDIR / "source"
sys.path.insert(0, str(SRC))

import config as cfg                     # noqa: E402
import fit as fitmod                     # noqa: E402
import kernels as kern                   # noqa: E402
from quadrature import make_quadrature, bregman_parts  # noqa: E402

_spec = importlib.util.spec_from_file_location("squared_gap", SQDIR / "squared_gap.py")
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

SIGMA = cfg.SIGMA
T_MAX = (0.6 * SIGMA) ** 2           # physical bound on t (as in the production chart)
BOX = 2.0                            # center bounds [-2, 2]^d (as in the production solver)
TAU_IN, TAU_OUT = (0.1 * SIGMA) ** 2, (0.2 * SIGMA) ** 2
M_PAIRS, S_SINGLE = 4, 2


class WorkLimit(Exception):
    pass


class Converged(Exception):
    pass


def source_hashes():
    paths = [Path(__file__), SQDIR / "squared_gap.py"] + sorted(SRC.glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Problem:
    """Target, quadrature, start and objective of one cohort case."""

    def __init__(self, d, geometry, seed, power=12, quadrature_seed=8128, reaction=1.0):
        self.d, self.geometry, self.seed = d, geometry, seed
        tb, tc = cfg.target_blocks(d, geometry)
        self.tb = [dict(b) for b in tb]
        for b in self.tb:
            b["c"] = np.asarray(b["c"], float)
            if b["kind"] == "pair":
                b["e"] = np.asarray(b["e"], float)
        self.tc = np.asarray(tc, float)
        # Production kernels for the target (completed pairs), as in run.Solver.
        self.target2 = float(self.tc @ kern.block_gram(self.tb).numpy() @ self.tc)
        X, W, _ = make_quadrature(np.array([b["c"] for b in self.tb]), power=power,
                                  seed=quadrature_seed + 100 * geometry + d)
        self.X = torch.as_tensor(X)
        self.W = reaction * np.asarray(W, float)
        self.U = kern.column_values(self.tb, self.X).numpy() @ self.tc
        starts = cfg.start_centers(d, seed, geometry)
        self.start_centers = starts
        beta, t, e = [], [], []
        for j in range(M_PAIRS):
            cp, cm = starts[2 * j], starts[2 * j + 1]
            diff = cp - cm
            h = 0.5 * float(np.linalg.norm(diff))
            beta.append(0.5 * (cp + cm))
            e.append(diff / (2 * h))
            t.append(h * h)
        self.z0 = dict(beta=np.array(beta), t=np.array(t), e=np.array(e),
                       c=np.array(starts[2 * M_PAIRS:]))
        self.fits = 0
        self.max_fits = None
        self.warm = None

    # ------------------------------------------------------------------ objective
    def blocks_from(self, beta, t, e_raw, c):
        en = e_raw / torch.linalg.vector_norm(e_raw, dim=1, keepdim=True)   # radial extension
        blocks = [dict(kind="pair", c=beta[j], e=en[j], t=t[j], h=0.0, completed=False) for j in range(M_PAIRS)]
        blocks += [dict(kind="single", c=c[k]) for k in range(S_SINGLE)]
        return blocks

    def evaluate(self, z, need_grad=True):
        """Return objective, canonical gradient (dict), coefficients and diagnostics at z.

        z = dict(beta=(m,d), t=(m,), e=(m,d) nonzero (normalized radially), c=(s,d)).
        """
        if self.max_fits is not None and self.fits >= self.max_fits:
            raise WorkLimit
        self.fits += 1
        beta = torch.tensor(z["beta"], requires_grad=need_grad)
        tt = torch.tensor(z["t"], requires_grad=need_grad)
        e = torch.tensor(z["e"], requires_grad=need_grad)
        c = torch.tensor(z["c"], requires_grad=need_grad)
        blocks = self.blocks_from(beta, tt, e, c)
        Mt = sg.block_gram(blocks)
        bt = sg.block_gram(blocks, self.tb) @ torch.as_tensor(self.tc)
        Pt = sg.column_values(blocks, self.X)
        M, b, Phi = (v.detach().numpy() for v in (Mt, bt, Pt))
        M = 0.5 * (M + M.T)
        g0 = self.warm if self.warm is not None else fitmod.solve_unconstrained_quadratic(M, b)
        g, info = fitmod.solve_coefficients(M / self.target2, b / self.target2, np.ones(len(b)), None,
                                            Phi, self.U, self.W / self.target2, g0)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("nonfinite coefficient fit")
        gap, residual, _ = bregman_parts(Phi @ g, self.U)
        obj = float((0.5 * g @ M @ g - b @ g + self.W @ gap) / self.target2)
        out = dict(obj=obj, coef=g, M=M, b=b, fit_info=info)
        if need_grad:
            gt = torch.as_tensor(g)
            proxy = (0.5 * gt @ Mt @ gt - bt @ gt
                     + torch.as_tensor(self.W * residual) @ (Pt @ gt)) / self.target2
            gb, gtt, ge, gc = torch.autograd.grad(proxy, [beta, tt, e, c])
            out["grad"] = dict(beta=gb.numpy(), t=gtt.numpy(), e=ge.numpy(), c=gc.numpy())
        return out

    def rel_error_double(self, ev):
        g, M, b = ev["coef"], ev["M"], ev["b"]
        return math.sqrt(max(g @ M @ g - 2 * b @ g + self.target2, 0.0) / self.target2)

    def endpoint_error(self, z, coef, dps=110):
        blocks = []
        for j in range(M_PAIRS):
            e = z["e"][j] / np.linalg.norm(z["e"][j])
            t = float(z["t"][j])
            if t == 0.0:
                blocks.append(dict(kind="pair", c=z["beta"][j], e=e, h=0.0, completed=True))
            else:
                blocks.append(dict(kind="pair", c=z["beta"][j], e=e, h=math.sqrt(t), completed=False))
        blocks += [dict(kind="single", c=z["c"][k]) for k in range(S_SINGLE)]
        return kern.high_precision_error(blocks, np.asarray(coef), self.tb, self.tc, dps=dps)


# ---------------------------------------------------------------------- helpers
def canonical_residual(z, grad):
    """Euclidean norm of R_1 (gamma = 1): box residuals of beta, t, c and tangent e-gradients."""
    rb = z["beta"] - np.clip(z["beta"] - grad["beta"], -BOX, BOX)
    rt = z["t"] - np.clip(z["t"] - grad["t"], 0.0, T_MAX)
    rc = z["c"] - np.clip(z["c"] - grad["c"], -BOX, BOX)
    re = grad["e"]          # radial extension: already tangent at unit e
    return float(math.sqrt((rb ** 2).sum() + (rt ** 2).sum() + (rc ** 2).sum() + (re ** 2).sum()))


def copyz(z):
    return {k: np.array(v, dtype=float, copy=True) for k, v in z.items()}


class Extremes:
    """Coordinatewise extremes of all evaluated trials (a posteriori search regions)."""

    def __init__(self):
        self.lo, self.hi, self.tmax, self.clipped = None, None, None, 0

    def add(self, z):
        cent = np.concatenate([z["beta"], z["c"]])
        if self.lo is None:
            self.lo, self.hi = cent.copy(), cent.copy()
            self.tmax = np.array(z["t"], float)
        else:
            self.lo, self.hi = np.minimum(self.lo, cent), np.maximum(self.hi, cent)
            self.tmax = np.maximum(self.tmax, z["t"])

    def separation(self):
        """Smallest margin dist(box_j, box_k) - sqrt(T_j) - sqrt(T_k) over block pairs, with
        T_j = max(observed t_j, 1.01 tau_out) for pairs and 0 for singletons."""
        nb = M_PAIRS + S_SINGLE
        rad = np.zeros(nb)
        rad[:M_PAIRS] = np.sqrt(np.maximum(self.tmax, 1.01 * TAU_OUT))
        margin = math.inf
        for j in range(nb):
            for k in range(j + 1, nb):
                gap = np.maximum(0.0, np.maximum(self.lo[j] - self.hi[k], self.lo[k] - self.hi[j]))
                dist = float(np.linalg.norm(gap))
                margin = min(margin, dist - rad[j] - rad[k])
        return margin


def state_summary(prob, z, ev):
    t = np.array(z["t"], float)
    coef = ev["coef"]
    raw = []
    for j in range(M_PAIRS):
        a, cc = coef[2 * j], coef[2 * j + 1]
        if t[j] > 0:
            h = math.sqrt(t[j])
            raw.extend([a / 2 + cc / (2 * h), a / 2 - cc / (2 * h)])
        else:
            raw.extend([math.nan, math.nan])
    raw.extend(coef[2 * M_PAIRS:])
    raw = np.array(raw)
    return dict(t=t.tolist(), coefficient_norm=float(np.linalg.norm(coef)),
                ordinary_weight_norm=(float(np.linalg.norm(raw)) if np.all(np.isfinite(raw)) else None))


# --------------------------------------------------------------- projected gradient
def run_pg(prob, chart, tol, max_fits, alpha0, beta_ls=0.5, eta=0.5, reversible=True):
    """Algorithm 4.1 (chart='t') or the same iteration in the half-gap (chart='h')."""
    prob.max_fits, prob.fits, prob.warm = max_fits, 0, None
    ext = Extremes()
    z = copyz(prob.z0)
    z["e"] = z["e"] / np.linalg.norm(z["e"], axis=1, keepdims=True)
    trace, events = [], []
    mode = ["stable" if z["t"][j] <= TAU_OUT else "ordinary" for j in range(M_PAIRS)]
    stored = [None] * M_PAIRS
    roundtrip_max = 0.0
    alphas = []
    last_row = None
    stop = "work_limit"
    t_start = time.perf_counter()
    try:
        ev = prob.evaluate(z)
        prob.warm = ev["coef"].copy()
        ext.add(z)
        it = 0
        while True:
            res = canonical_residual(z, ev["grad"])
            row = dict(iteration=it, fits=prob.fits, objective=ev["obj"], residual=res,
                       rel_error_double=prob.rel_error_double(ev), alpha=(alphas[-1] if alphas else None),
                       **state_summary(prob, z, ev))
            last_row = row
            if it < 100 or it % 10 == 0:
                trace.append(row)
            if res <= tol:
                stop = "tolerance"
                break
            # Reversible storage: recover canonical geometry from ordinary storage (Algorithm 4.1, line 3).
            roundtrip_now = 0.0
            if chart == "t" and reversible:
                for j in range(M_PAIRS):
                    if mode[j] == "ordinary":
                        (bp, bm, wp, wm) = stored[j]
                        beta = 0.5 * (bp + bm)
                        h = 0.5 * float(np.linalg.norm(bp - bm))
                        e = (bp - bm) / (2 * h)
                        diff = max(float(np.max(np.abs(beta - z["beta"][j]))), abs(h * h - z["t"][j]),
                                   float(np.max(np.abs(e - z["e"][j]))))
                        roundtrip_now = max(roundtrip_now, diff)
                        roundtrip_max = max(roundtrip_max, diff)
                        z["beta"][j], z["t"][j], z["e"][j] = beta, h * h, e
                # In exact arithmetic the recovered geometry equals z_k; the accepted evaluation is
                # reused unless rounding moved the state by more than 1e-12 (then it is refitted).
                if any(m == "ordinary" for m in mode) and roundtrip_now > 1e-12:
                    ext.add(z)
                    ev = prob.evaluate(z)
                    prob.warm = ev["coef"].copy()
            g = ev["grad"]
            if chart == "t":
                gx = dict(beta=g["beta"], t=g["t"], e=g["e"], c=g["c"])
                x_t = z["t"]
            else:
                h_now = np.sqrt(z["t"])
                gx = dict(beta=g["beta"], t=2 * h_now * g["t"], e=g["e"], c=g["c"])
                x_t = h_now
            alpha = alpha0
            accepted = False
            while alpha > 1e-20:
                trial = dict(beta=np.clip(z["beta"] - alpha * gx["beta"], -BOX, BOX),
                             c=np.clip(z["c"] - alpha * gx["c"], -BOX, BOX))
                ee = z["e"] - alpha * gx["e"]
                trial["e"] = ee / np.linalg.norm(ee, axis=1, keepdims=True)
                if chart == "t":
                    tnew = np.clip(x_t - alpha * gx["t"], 0.0, T_MAX)
                    trial["t"] = tnew
                    step2 = ((trial["beta"] - z["beta"]) ** 2).sum() + ((tnew - x_t) ** 2).sum() \
                        + ((trial["c"] - z["c"]) ** 2).sum() + ((trial["e"] - z["e"]) ** 2).sum()
                else:
                    hnew = np.clip(x_t - alpha * gx["t"], 0.0, math.sqrt(T_MAX))
                    trial["t"] = hnew * hnew
                    step2 = ((trial["beta"] - z["beta"]) ** 2).sum() + ((hnew - x_t) ** 2).sum() \
                        + ((trial["c"] - z["c"]) ** 2).sum() + ((trial["e"] - z["e"]) ** 2).sum()
                if (np.any(np.abs(z["beta"] - alpha * gx["beta"]) > BOX)
                        or np.any(np.abs(z["c"] - alpha * gx["c"]) > BOX)):
                    ext.clipped += 1
                ext.add(trial)
                evt = prob.evaluate(trial)
                if evt["obj"] <= ev["obj"] - eta / (2 * alpha) * step2:
                    accepted = True
                    break
                alpha *= beta_ls
            if not accepted:
                stop = "line_search_failure"
                break
            alphas.append(alpha)
            z, ev = trial, evt
            prob.warm = ev["coef"].copy()
            it += 1
            # Reversible storage rule (Algorithm 4.1, line 6).
            if chart == "t" and reversible:
                for j in range(M_PAIRS):
                    t = float(z["t"][j])
                    if mode[j] == "stable" and t >= TAU_OUT:
                        h = math.sqrt(t)
                        a, cc = ev["coef"][2 * j], ev["coef"][2 * j + 1]
                        stored[j] = (z["beta"][j] + h * z["e"][j], z["beta"][j] - h * z["e"][j],
                                     a / 2 + cc / (2 * h), a / 2 - cc / (2 * h))
                        mode[j] = "ordinary"
                        events.append(dict(iteration=it, block=j, change="stable->ordinary", t=t))
                    elif mode[j] == "ordinary" and t <= TAU_IN:
                        mode[j] = "stable"
                        stored[j] = None
                        events.append(dict(iteration=it, block=j, change="ordinary->stable", t=t))
                    elif mode[j] == "ordinary":
                        h = math.sqrt(t)
                        a, cc = ev["coef"][2 * j], ev["coef"][2 * j + 1]
                        stored[j] = (z["beta"][j] + h * z["e"][j], z["beta"][j] - h * z["e"][j],
                                     a / 2 + cc / (2 * h), a / 2 - cc / (2 * h))
    except WorkLimit:
        stop = "work_limit"
    seconds = time.perf_counter() - t_start
    if trace[-1] is not last_row:
        trace.append(last_row)
    return dict(z=z, ev=ev, trace=trace, events=events, stop=stop, seconds=seconds, ext=ext,
                alphas=alphas, roundtrip_max=roundtrip_max, switches=len(events))


# ------------------------------------------------------------------- L-BFGS-B
def run_lbfgs(prob, chart, tol, max_fits, maxcor=20, maxls=40, max_stalls=2):
    prob.max_fits, prob.fits, prob.warm = max_fits, 0, None
    ext = Extremes()
    z = copyz(prob.z0)
    z["e"] = z["e"] / np.linalg.norm(z["e"], axis=1, keepdims=True)
    d = prob.d
    trace, returns = [], []
    stop = "work_limit"
    t_start = time.perf_counter()
    state = dict(z=z, ev=None, it=0, best_res=math.inf)

    def pack(zz, frames):
        tpart = zz["t"] if chart == "t" else np.sqrt(zz["t"])
        return np.concatenate([zz["beta"].ravel(), tpart, np.zeros(M_PAIRS * (d - 1)), zz["c"].ravel()])

    def unpack(p, base_e, frames):
        k = 0
        beta = p[k:k + M_PAIRS * d].reshape(M_PAIRS, d); k += M_PAIRS * d
        tp = p[k:k + M_PAIRS]; k += M_PAIRS
        pd = p[k:k + M_PAIRS * (d - 1)].reshape(M_PAIRS, d - 1); k += M_PAIRS * (d - 1)
        c = p[k:].reshape(S_SINGLE, d)
        e_raw = np.stack([base_e[j] + frames[j] @ pd[j] for j in range(M_PAIRS)])
        t = tp if chart == "t" else tp * tp
        return dict(beta=beta.copy(), t=np.array(t, float), e=e_raw, c=c.copy()), tp

    cache = {}

    def fun(p, base_e, frames):
        key = p.tobytes()
        if key in cache:
            return cache[key][0], cache[key][1]
        zz, tp = unpack(p, base_e, frames)
        ext.add(dict(beta=zz["beta"], c=zz["c"], t=zz["t"]))
        ev = prob.evaluate(zz)
        g = ev["grad"]
        gt = g["t"] if chart == "t" else 2 * tp * g["t"]
        gpd = np.stack([frames[j].T @ g["e"][j] for j in range(M_PAIRS)])
        grad = np.concatenate([g["beta"].ravel(), gt, gpd.ravel(), g["c"].ravel()])
        zn = dict(beta=zz["beta"], t=zz["t"], e=zz["e"] / np.linalg.norm(zz["e"], axis=1, keepdims=True), c=zz["c"])
        # Canonical gradient at the normalized state: the radial extension gives the same tangent
        # e-gradient up to the factor |e_raw| >= 1; recompute it exactly from the unnormalized one.
        gcan = dict(beta=g["beta"], t=g["t"], c=g["c"],
                    e=g["e"] * np.linalg.norm(zz["e"], axis=1, keepdims=True))
        cache.clear()
        cache[key] = (ev["obj"], grad, zn, ev, gcan)
        return ev["obj"], grad

    try:
        stalls = 0
        while True:
            base_e = state["z"]["e"] / np.linalg.norm(state["z"]["e"], axis=1, keepdims=True)
            frames = [null_space(base_e[j].reshape(1, d)) for j in range(M_PAIRS)]
            p0 = pack(state["z"], frames)
            ub_t = T_MAX if chart == "t" else math.sqrt(T_MAX)
            bounds = ([(-BOX, BOX)] * (M_PAIRS * d) + [(0.0, ub_t)] * M_PAIRS
                      + [(-2.0, 2.0)] * (M_PAIRS * (d - 1)) + [(-BOX, BOX)] * (S_SINGLE * d))
            if state["ev"] is None:
                fun(p0, base_e, frames)
                _, _, zn, ev, gcan = cache[p0.tobytes()]
                prob.warm = ev["coef"].copy()
                state.update(z=zn, ev=ev)
                res = canonical_residual(zn, gcan)
                state["best_res"] = res
                trace.append(dict(iteration=0, fits=prob.fits, objective=ev["obj"], residual=res,
                                  rel_error_double=prob.rel_error_double(ev), **state_summary(prob, zn, ev)))
                if res <= tol:
                    stop = "tolerance"
                    raise Converged

            def callback(xk):
                key = xk.tobytes()
                if key not in cache:
                    fun(xk, base_e, frames)
                _, _, zn, ev, gcan = cache[key]
                prob.warm = ev["coef"].copy()
                state["it"] += 1
                state.update(z=zn, ev=ev)
                res = canonical_residual(zn, gcan)
                trace.append(dict(iteration=state["it"], fits=prob.fits, objective=ev["obj"], residual=res,
                                  rel_error_double=prob.rel_error_double(ev), **state_summary(prob, zn, ev)))
                if res <= tol:
                    raise Converged

            res_before = trace[-1]["residual"]
            try:
                result = minimize(lambda p: fun(p, base_e, frames), p0, jac=True, method="L-BFGS-B",
                                  bounds=bounds, callback=callback,
                                  options=dict(ftol=0.0, gtol=0.0, maxcor=maxcor, maxls=maxls,
                                               maxiter=100000, maxfun=100000))
            except Converged:
                stop = "tolerance"
                raise
            returns.append(dict(fits=prob.fits, status=int(result.status), message=str(result.message),
                                residual=trace[-1]["residual"]))
            # Continue from the last accepted iterate (callback state), never from a rejected trial.
            if trace[-1]["residual"] < 0.9 * res_before:
                stalls = 0
            else:
                stalls += 1
            if stalls >= max_stalls:
                stop = "stalled:" + str(result.message)
                break
    except Converged:
        stop = "tolerance"
    except WorkLimit:
        stop = "work_limit"
    seconds = time.perf_counter() - t_start
    return dict(z=state["z"], ev=state["ev"], trace=trace, events=[], stop=stop, seconds=seconds,
                ext=ext, returns=returns, switches=0)


# ------------------------------------------------------------------- one run
def run_one(d, geometry, seed, arm, tol=1e-7, max_fits=3000, alpha0=1.0, dps=110):
    prob = Problem(d, geometry, seed)
    if arm in ("pg_t", "pg_h"):
        out = run_pg(prob, arm[-1], tol, max_fits, alpha0)
    elif arm in ("lbfgs_t", "lbfgs_h"):
        out = run_lbfgs(prob, arm[-1], tol, max_fits)
    else:
        raise ValueError(arm)
    z, ev = out["z"], out["ev"]
    t0 = time.perf_counter()
    hp = prob.endpoint_error(z, ev["coef"], dps=dps)
    audit = time.perf_counter() - t0
    tr = out["trace"]
    final = tr[-1]
    record = dict(
        d=d, geometry=geometry, seed=seed, arm=arm, tol=tol, max_fits=max_fits,
        alpha0=(alpha0 if arm.startswith("pg") else None),
        stop=out["stop"], fits=prob.fits, iterations=final["iteration"],
        final_residual=final["residual"], final_objective=final["objective"],
        final_relative_error_110=float(hp["relative"]),
        final_relative_error_double=final["rel_error_double"],
        final_t=final["t"], zero_gaps=int(sum(1 for t in final["t"] if t == 0.0)),
        coefficient_norm=final["coefficient_norm"], ordinary_weight_norm=final["ordinary_weight_norm"],
        switches=out.get("switches", 0), storage_events=out.get("events", []),
        roundtrip_max=out.get("roundtrip_max"),
        separation_margin=out["ext"].separation(), box_clipped_trials=out["ext"].clipped,
        trial_t_max=out["ext"].tmax.tolist(),
        alphas_summary=(dict(min=float(np.min(out["alphas"])), max=float(np.max(out["alphas"])),
                             backtracked=int(np.sum(np.array(out["alphas"]) < alpha0)))
                        if out.get("alphas") else None),
        lbfgs_returns=out.get("returns"),
        solve_seconds=out["seconds"], audit_seconds=audit,
        trace=tr, final_state=dict(beta=z["beta"].tolist(), t=z["t"].tolist(), e=z["e"].tolist(),
                                   c=z["c"].tolist(), coef=ev["coef"].tolist()),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, platform=platform.platform()),
        sources=source_hashes(),
        settings=dict(sigma=SIGMA, T_max=T_MAX, box=BOX, tau_in=TAU_IN, tau_out=TAU_OUT,
                      quadrature_power=12, quadrature_seed=8128, reaction=1.0,
                      pg=dict(beta=0.5, eta=0.5), lbfgs=dict(maxcor=20, maxls=40, ftol=0.0, gtol=0.0,
                                                             restart_stalls=2)))
    return record


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--geometry", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["pg_t", "pg_h", "lbfgs_t", "lbfgs_h"])
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-fits", type=int, default=3000)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    rec = run_one(a.d, a.geometry, a.seed, a.arm, a.tol, a.max_fits, a.alpha0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, allow_nan=True) + "\n")
    tmp.replace(a.output)
    print(json.dumps({k: rec[k] for k in ("d", "geometry", "seed", "arm", "stop", "fits", "iterations",
                                           "final_residual", "final_relative_error_110", "zero_gaps",
                                           "separation_margin", "solve_seconds")}))
