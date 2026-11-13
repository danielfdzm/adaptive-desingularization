"""Adaptive desingularization with unconstrained coefficients (one space dimension).

Implements the algorithm of the draft (Section 2.4): stable Newton blocks, coefficient fitting
by direct unconstrained solves, variable-projection center optimization, collision detection
with persistence, and centroid moment/Taylor replacement with an energy gate.
Residual-driven insertion and pruning are optional extensions, disabled by default.
The energy is the quadratic H^1(R) energy of
-u'' + u = f with manufactured f = A u*, so all pairings are analytic.

Also contains the "standard optimization" baseline: ordinary Gaussians with unconstrained
least-squares weights and L-BFGS-B over the centers (variable projection).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar, OptimizeResult

from gauss1d import (AtomTable, Block, Kernel1D, expansion_matrix, expansion_matrix_derivative,
                     needed_extra_order, newton_union, raw_weights_of_block, shift_atoms)
from coefficient_fit import solve_unconstrained_qp


# ----------------------------------------------------------------------------
# Problem: quadratic energy with a manufactured Gaussian-derivative solution
# ----------------------------------------------------------------------------

class Problem1D:
    """J(v) = 1/2 ||v||_H^2 - <f, v>,  f = A u*,  u* given as an atom table (distinct centers)."""

    def __init__(self, kern: Kernel1D, target: AtomTable, rho_fn, C=None, L0: float = 5.0, rho_len: int = 60):
        self.kern = kern
        self.target = target
        self.rho_fn = rho_fn
        self.rho = np.array([rho_fn(r) for r in range(rho_len)])
        if C is not None:
            raise ValueError("Coefficient budgets are no longer supported")
        self.C = None
        self.L0 = L0
        self.target_norm2 = kern.norm2(target)
        self.target_norm = math.sqrt(self.target_norm2)
        self.target_centers = np.unique(target.centers)
        self.J_target = -0.5 * self.target_norm2

    # -- cancellation-free difference u* - v ----------------------------------
    def difference_table(self, atoms: AtomTable) -> AtomTable:
        """Atom table of u* - v in which every atom of v lying within 0.5 sigma of a target center is
        re-expanded at that center, so that coefficients (not function values) cancel."""
        kern = self.kern
        far_c, far_o, far_a = [], [], []
        groups = {}
        for c, s, a in zip(atoms.centers, atoms.orders, atoms.coeffs):
            if a == 0.0:
                continue
            i = int(np.argmin(np.abs(self.target_centers - c)))
            if abs(self.target_centers[i] - c) <= 0.5 * kern.sigma:
                groups.setdefault(i, []).append((c, int(s), a))
            else:
                far_c.append(c); far_o.append(int(s)); far_a.append(-a)
        coef = {}
        for i, lst in groups.items():
            tab = AtomTable(np.array([x[0] for x in lst]), np.array([x[1] for x in lst]), np.array([x[2] for x in lst]))
            sh = shift_atoms(kern, tab, float(self.target_centers[i]))
            for s, a in zip(sh.orders, sh.coeffs):
                coef[(i, int(s))] = coef.get((i, int(s)), 0.0) - a
        for c, s, a in zip(self.target.centers, self.target.orders, self.target.coeffs):
            i = int(np.argmin(np.abs(self.target_centers - c)))
            coef[(i, int(s))] = coef.get((i, int(s)), 0.0) + a
        cen = [self.target_centers[i] for (i, s) in coef] + far_c
        ords = [s for (i, s) in coef] + far_o
        cof = [coef[k] for k in coef] + far_a
        return AtomTable(np.array(cen, dtype=float), np.array(ords, dtype=int), np.array(cof, dtype=float))

    def error_norm(self, atoms: AtomTable) -> float:
        d = self.difference_table(atoms)
        return math.sqrt(max(self.kern.norm2(d), 0.0))

    def rel_error(self, atoms: AtomTable) -> float:
        return self.error_norm(atoms) / self.target_norm

    _quad_grid = None

    def error_norm_quad(self, atoms: AtomTable, L: float = 12.0, h: float = 0.005) -> float:
        """||v - u*||_H by the trapezoid rule on [-L, L] from pointwise values (independent evaluation;
        spectrally accurate for these Gaussian integrands)."""
        if Problem1D._quad_grid is None or Problem1D._quad_grid[0] != (L, h):
            x = np.arange(-L, L + 0.5 * h, h)
            Problem1D._quad_grid = ((L, h), x)
        x = Problem1D._quad_grid[1]
        d0 = atoms.evaluate(self.kern, x) - self.target.evaluate(self.kern, x)
        shifted = AtomTable(atoms.centers, atoms.orders + 1, atoms.coeffs)
        tshift = AtomTable(self.target.centers, self.target.orders + 1, self.target.coeffs)
        d1 = shifted.evaluate(self.kern, x) - tshift.evaluate(self.kern, x)
        return math.sqrt(h * float(np.sum(d0 * d0 + d1 * d1)))

    def rel_error_quad(self, atoms: AtomTable) -> float:
        return self.error_norm_quad(atoms) / self.target_norm

    def J(self, atoms: AtomTable) -> float:
        """Energy  1/2||v||^2 - (u*, v)  (its minimum is J_target = -1/2||u*||^2)."""
        return 0.5 * self.kern.norm2(atoms) - self.kern.inner(atoms, self.target)

    def excess(self, atoms: AtomTable) -> float:
        """J(v) - J(u*) = 1/2 ||v - u*||^2, evaluated without cancellation."""
        return 0.5 * self.error_norm(atoms) ** 2

    def residual_atoms(self, atoms: AtomTable) -> AtomTable:
        return self.difference_table(atoms)

    def correlations(self, resid: AtomTable, r: int, b):
        """q_r(b) = (u* - v, G^{(r)}(.-b))_H for an array of centers b."""
        b = np.asarray(b, dtype=float)
        out = np.zeros_like(b)
        for c, s, a in zip(resid.centers, resid.orders, resid.coeffs):
            if a != 0.0:
                out += a * ((-1.0) ** s) * self.kern.K_n(int(s) + r, c - b)
        return out


def default_rho_fn(kern: Kernel1D, s: float = 1.0):
    return lambda r: (1.0 + r) ** s * kern.kappa(r)


def tailored_rho_fn(kern: Kernel1D, k: int):
    return lambda r: (1.0 + abs(r - k)) * kern.kappa(r)


def stable_cost(target: AtomTable, rho: np.ndarray) -> float:
    return float(sum(rho[int(r)] * abs(a) for r, a in zip(target.orders, target.coeffs)))


# ----------------------------------------------------------------------------
# Adaptive solver
# ----------------------------------------------------------------------------

@dataclass
class AdaptiveConfig:
    N: int = 6                       # multiplicity capacity
    d_watch: float = 1.0             # watch radius (group extent) in units of sigma
    d_on: float = 0.5                # activation radius in units of sigma
    d_split: float = 1.25            # split a Newton block back when extent > d_split*d_watch*sigma
    kappa_on: float = 0.1            # Gram diagnostic threshold
    persistence: int = 2             # consecutive scans
    scan_period: int = 20            # L-BFGS iterations between scans
    tau_rel: float = 1e-2            # state tolerance for a transfer, relative to ||v||
    tau_abs: float = 0.0
    eps_tot: float = 1e-3            # total energy allowance, relative to ||u*||^2
    completion: bool = True
    insertion: bool = False        # optional extension; not part of the default method
    insert_rule: str = "per_slot"    # "per_slot": largest measured decrease per slot; "decrease": largest decrease; "correlation": largest one-feature line decrease
    r_max_insert: int = 4            # highest derivative order proposed by insertion
    insert_min_decrease: float = 1e-12   # relative energy decrease below which insertion stops
    trace_residual: bool = False     # optional residual candidate diagnostics
    prune: bool = False             # optional extension used with insertion
    prune_rel: float = 1e-6          # state tolerance for pruning vanishing trailing orders (relative to ||v||)
    grid_L: float = 12.0             # residual search interval [-L, L]
    grid_n: int = 2401
    max_fits: int = 3000
    lbfgs_maxiter_total: int = 6000
    lbfgs_ftol: float = 1e-15
    lbfgs_gtol: float = 1e-12
    verbose: bool = False


@dataclass
class Trace:
    fits: list = field(default_factory=list)
    rel_err: list = field(default_factory=list)
    rel_err_quad: list = field(default_factory=list)   # independent trapezoid evaluation
    energy: list = field(default_factory=list)
    coef_norm: list = field(default_factory=list)      # stable (Newton) coefficient norm
    raw_norm: list = field(default_factory=list)       # ordinary-coordinate equivalent where defined
    min_halfgap: list = field(default_factory=list)
    cost: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    events: list = field(default_factory=list)         # (fits, kind, dict)
    gaps: list = field(default_factory=list)           # dicts


class AdaptiveSolver1D:
    def __init__(self, prob: Problem1D, blocks: list, cfg: AdaptiveConfig):
        self.prob = prob
        self.kern = prob.kern
        self.cfg = cfg
        self.blocks = [b.copy() for b in blocks]
        self.rho = prob.rho
        self.C = prob.C
        self.n_fits = 0
        self.n_scans = 0
        self.j_allow = 0                    # index of the next energy allowance
        self.eps_tot = cfg.eps_tot * prob.target_norm2
        self.trace = Trace()
        self.flag_counters = {}
        self.gamma_prev = None
        self.best_rel = np.inf
        self.stop_reason = None
        self.unresolved_flags = 0
        self.regroup_failures = 0
        self.transfer_rejections = 0
        self.n_completions = 0
        self.n_insertions = 0
        self.last_cert = None
        self.lbfgs_iters = 0
        self.abnormal_returns = 0
        self.normal_return = True

    # ---- bookkeeping ---------------------------------------------------------
    @property
    def slots(self) -> int:
        return sum(b.nu for b in self.blocks)

    def cost(self, blocks=None) -> float:
        blocks = self.blocks if blocks is None else blocks
        return sum(b.cost(self.rho) for b in blocks)

    def atoms(self, blocks=None) -> AtomTable:
        blocks = self.blocks if blocks is None else blocks
        return AtomTable.concat([b.atoms(self.kern) for b in blocks])

    def coef_norm(self) -> float:
        return math.sqrt(sum(float(b.gamma @ b.gamma) for b in self.blocks))

    def raw_norm(self) -> float:
        tot = 0.0
        for b in self.blocks:
            if (not b.confluent) and np.all(b.mult == 1):
                w = raw_weights_of_block(b)
                tot += float(w @ w)
            else:
                tot += float(b.gamma @ b.gamma)
        return math.sqrt(tot)

    def min_halfgap(self) -> float:
        nodes = np.sort(np.concatenate([b.nodes for b in self.blocks]))
        if len(nodes) < 2:
            return np.inf
        return float(np.diff(nodes).min() / 2.0)

    def next_allowance(self) -> float:
        return self.eps_tot * 2.0 ** (-self.j_allow - 1)

    def record(self):
        at = self.atoms()
        rel = self.prob.rel_error(at)
        self.best_rel = min(self.best_rel, rel)
        tr = self.trace
        tr.fits.append(self.n_fits)
        tr.rel_err.append(rel)
        tr.rel_err_quad.append(self.prob.rel_error_quad(at))
        tr.energy.append(self.prob.J(at))
        tr.coef_norm.append(self.coef_norm())
        tr.raw_norm.append(self.raw_norm())
        tr.min_halfgap.append(self.min_halfgap())
        tr.cost.append(self.cost())
        tr.slots.append(self.slots)
        if self.cfg.trace_residual:
            self.residual_search()

    def log(self, *a):
        if self.cfg.verbose:
            print(*a)

    # ---- parameters <-> blocks ------------------------------------------------
    def pack(self) -> np.ndarray:
        return np.concatenate([b.nodes for b in self.blocks])

    def unpack(self, p: np.ndarray) -> list:
        out = []
        i = 0
        for b in self.blocks:
            nb = b.copy()
            nb.nodes = p[i:i + len(b.nodes)].copy()
            i += len(b.nodes)
            out.append(nb)
        return out

    # ---- assembly ------------------------------------------------------------
    def assemble(self, blocks):
        tables, Ts, slices, rhos = [], [], [], []
        off = 0
        for b in blocks:
            beta, T = b.expansion(self.kern)
            S = T.shape[0]
            tables.append(AtomTable(np.full(S, beta), np.arange(S), T @ b.gamma))
            Ts.append(T)
            slices.append(slice(off, off + S))
            off += S
            rhos.append(self.rho[: b.nu])
        A = AtomTable.concat(tables)
        return A, Ts, slices, np.concatenate(rhos)

    def fit(self, blocks, warm=None):
        A, Ts, slices, rho_cols = self.assemble(blocks)
        Gall = self.kern.gram(A)
        gf = self.kern.gram(A, self.prob.target) @ self.prob.target.coeffs
        ncol = sum(b.nu for b in blocks)
        Tbig = np.zeros((len(A), ncol))
        c0 = 0
        for T, sl in zip(Ts, slices):
            Tbig[sl, c0:c0 + T.shape[1]] = T
            c0 += T.shape[1]
        M = Tbig.T @ Gall @ Tbig
        M = 0.5 * (M + M.T)
        bvec = Tbig.T @ gf
        self.n_fits += 1
        gamma, info = solve_unconstrained_qp(M, bvec)
        return gamma, M, bvec, A, Tbig, slices, Gall, gf, info

    def set_gamma(self, blocks, gamma):
        c0 = 0
        for b in blocks:
            b.gamma = gamma[c0:c0 + b.nu].copy()
            c0 += b.nu

    # ---- variable projection objective ----------------------------------------
    def objective(self, p: np.ndarray):
        blocks = self.unpack(p)
        gamma, M, bvec, A, Tbig, slices, Gall, gf, info = self.fit(blocks, warm=self.gamma_prev)
        self.gamma_prev = gamma
        J = 0.5 * gamma @ M @ gamma - bvec @ gamma
        c = Tbig @ gamma
        resid = Gall @ c - gf            # (v - u*, atom_i)_H
        grad = np.zeros_like(p)
        i = 0
        c0 = 0
        for b, sl in zip(blocks, slices):
            g_b = gamma[c0:c0 + b.nu]
            Tb = Tbig[sl, c0:c0 + b.nu]
            if b.confluent:
                cb = Tb @ g_b
                dc = np.zeros(len(cb))
                dc[1:] = -cb[:-1]
                grad[i] = dc @ resid[sl]
                i += 1
            else:
                beta = b.beta
                tn = b.all_nodes()
                K = Tb.shape[0] - b.nu
                jj = 0
                for m in b.mult:
                    dc = np.zeros(Tb.shape[0])
                    for rep in range(m):
                        dT = expansion_matrix_derivative(tn, beta, K, jj)
                        dc += dT @ g_b
                        jj += 1
                    grad[i] = dc @ resid[sl]
                    i += 1
            c0 += b.nu
        return J, grad

    def optimize_centers(self, maxiter: int):
        p0 = self.pack()
        bounds = [(-self.prob.L0, self.prob.L0)] * len(p0)
        try:
            res = minimize(self.objective, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter, "ftol": self.cfg.lbfgs_ftol, "gtol": self.cfg.lbfgs_gtol, "maxls": 40})
            self.lbfgs_iters += res.nit
            blocks = self.unpack(res.x)
            gamma, *_ = self.fit(blocks, warm=self.gamma_prev)
            self.set_gamma(blocks, gamma)
            self.gamma_prev = gamma
            self.blocks = blocks
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            # Keep the last fitted state; never substitute a regularized solve.
            res=OptimizeResult(x=p0,nit=0,status=2,success=False,message=str(exc))
            self.gamma_prev=None
        self.record()
        if res.status == 2:
            self.abnormal_returns += 1
        return res

    # ---- detection -------------------------------------------------------------
    def dictionary_columns(self, group_blocks) -> AtomTable:
        """Hermite dictionary columns G^{(i)}(.-c), i < multiplicity, over all distinct nodes of the group."""
        cen, ords = [], []
        for b in group_blocks:
            for c, m in zip(b.nodes, b.mult):
                for i in range(int(m)):
                    cen.append(c); ords.append(i)
        return AtomTable(np.array(cen), np.array(ords, dtype=int), np.ones(len(cen)))

    def diagnostics(self, group_blocks):
        """Square root of the smallest normalized Gram eigenvalue."""
        cols = self.dictionary_columns(group_blocks)
        G = self.kern.gram(cols)
        s = np.sqrt(np.diag(G))
        Gn = G / s[:, None] / s[None, :]
        lam_min = float(np.linalg.eigvalsh(Gn).min())
        return math.sqrt(max(lam_min, 0.0))

    def groups(self):
        order = np.argsort([b.beta for b in self.blocks])
        bl = [self.blocks[i] for i in order]
        ids = [int(i) for i in order]
        thr = self.cfg.d_watch * self.kern.sigma
        groups = []
        i = 0
        n = len(bl)
        while i < n:
            lo = bl[i].nodes.min(); hi = bl[i].nodes.max()
            j = i
            while j + 1 < n:
                lo2 = min(lo, bl[j + 1].nodes.min()); hi2 = max(hi, bl[j + 1].nodes.max())
                if hi2 - lo2 <= thr:
                    lo, hi, j = lo2, hi2, j + 1
                else:
                    break
            members = ids[i:j + 1]
            if len(members) >= 2 or (not bl[i].confluent):
                groups.append((members, float(hi - lo)))
            i = j + 1
        return groups

    def regroup(self, members) -> bool:
        """Exact Newton change of coordinates; coefficient size is diagnostic only."""
        blocks = [self.blocks[i] for i in members]
        new = newton_union(self.kern, blocks)
        old_cost = self.cost(blocks)
        rest = [b for k, b in enumerate(self.blocks) if k not in members]
        self.trace.events.append((self.n_fits, "regroup", {
            "nu": new.nu, "extent": new.diameter,
            "cost_group_old": old_cost, "cost_group_new": new.cost(self.rho), "exact": True}))
        self.blocks = rest + [new]
        self.gamma_prev = None
        return True

    def split(self, index: int) -> bool:
        """Inverse change of coordinates: a non-confluent block back to one confluent block per node."""
        b = self.blocks[index]
        if b.confluent:
            return False
        cols = self.dictionary_columns([b])
        G = self.kern.gram(cols)
        s = np.sqrt(np.diag(G))
        vB = b.atoms(self.kern)
        g = (self.kern.gram(cols, vB) @ vB.coeffs) / s
        x = np.linalg.solve(G / s[:, None] / s[None, :], g) / s      # Hermite data coefficients on G^{(i)}_c
        singles = []
        k = 0
        for c, m in zip(b.nodes, b.mult):
            coeffs = x[k:k + m]
            gam = np.array([((-1.0) ** i) * coeffs[i] for i in range(m)])
            singles.append(Block([c], gam, [m]))
            k += m
        rest = [bb for kk, bb in enumerate(self.blocks) if kk != index]
        cost_new = self.cost(rest) + sum(sb.cost(self.rho) for sb in singles)
        err = math.sqrt(max(self.kern.norm2(vB.minus(AtomTable.concat([sb.atoms(self.kern) for sb in singles]))), 0.0))
        self.trace.events.append((self.n_fits, "split", {"nu": b.nu, "extent": b.diameter, "cost": cost_new, "err": err}))
        self.log(f"  split block nodes {np.round(b.nodes,4)} mult {b.mult} extent={b.diameter:.3e}")
        self.blocks = rest + singles
        self.gamma_prev = None
        return True

    def transfer_trial(self, block: Block):
        """Moment/Taylor replacement at the multiplicity-weighted centroid."""
        beta = block.beta
        nu = block.nu
        tn = block.all_nodes()
        delta = float(np.max(np.abs(beta - tn)))
        K = needed_extra_order(self.kern, delta, nu)
        T = expansion_matrix(tn, beta, K)
        c = T @ block.gamma
        kap = self.kern.kappas(nu + K + 1)
        trunc = sum(abs(block.gamma[r]) * (delta ** (K + 1 + nu - r)) / math.factorial(K + 1 + nu - r) * kap[nu + K + 1] for r in range(nu))
        # Keep the expansion coefficients c_s for s < nu.
        gam_tay = np.array([((-1.0) ** s) * c[s] for s in range(nu)])
        tail = AtomTable(np.full(len(c) - nu, beta), np.arange(nu, len(c)), c[nu:])
        e_val = math.sqrt(max(self.kern.norm2(tail), 0.0)) * (1 + 1e-10) + 2 * trunc
        e_an = kap[nu] * sum(abs(block.gamma[r]) * (delta ** (nu - r)) / math.factorial(nu - r) for r in range(nu))
        return {"kind": "taylor", "gamma_new": gam_tay, "e_val": e_val, "e_an": e_an, "beta": beta}

    def attempt_completion(self, block_index: int) -> bool:
        block = self.blocks[block_index]
        if block.confluent:
            return False
        at = self.atoms()
        v_norm = math.sqrt(max(self.kern.norm2(at), 0.0))
        B_v = self.prob.error_norm(at)            # ||f - A v||_{H^-1} = ||u* - v||_H for this load
        J_before = self.prob.J(at)
        eps_j = self.next_allowance()
        tol_state = self.cfg.tau_abs + self.cfg.tau_rel * v_norm
        tr = self.transfer_trial(block)
        e = tr["e_val"]
        gate = B_v * e + 0.5 * e * e
        if e <= tol_state and gate <= eps_j:
            new = Block([tr["beta"]], tr["gamma_new"], [block.nu])
            blocks_new = [b for k, b in enumerate(self.blocks) if k != block_index] + [new]
            J_trial = self.prob.J(self.atoms(blocks_new))
            gamma, *_ = self.fit(blocks_new, warm=None)
            self.set_gamma(blocks_new, gamma)
            J_after = self.prob.J(self.atoms(blocks_new))
            self.blocks = blocks_new
            self.gamma_prev = None
            self.j_allow += 1
            self.n_completions += 1
            rel_after = self.prob.rel_error(self.atoms())
            self.trace.events.append((self.n_fits, "completion", {
                "before_block": {"nodes": block.nodes.copy(), "mult": block.mult.copy(), "gamma": block.gamma.copy()},
                "trial_block": {"nodes": [tr["beta"]], "mult": [block.nu], "gamma": tr["gamma_new"].copy()},
                "state_tolerance": tol_state,
                "transfer": tr["kind"], "nu": block.nu, "beta": tr["beta"], "extent": block.diameter,
                "e_val": e, "e_an": tr["e_an"], "gate": gate, "eps_j": eps_j, "B_v": B_v,
                "dJ_transfer": J_trial - J_before, "dJ_after_refit": J_after - J_before,
                "rel_err_before": B_v / self.prob.target_norm, "rel_err_after": rel_after}))
            self.log(f"  completion ({tr['kind']}) nu={block.nu} extent={block.diameter:.2e} e={e:.2e} (a priori {tr['e_an']:.2e}) "
                     f"gate={gate:.2e}<=eps_j={eps_j:.2e} dJ={J_trial-J_before:+.2e} -> after refit {J_after-J_before:+.2e}; rel err {B_v/self.prob.target_norm:.2e} -> {rel_after:.2e}")
            self.record()
            return True
        self.transfer_rejections += 1
        self.trace.events.append((self.n_fits, "transfer_rejected", {"nu": block.nu, "extent": block.diameter,
                                                                      "e_val": e,
                                                                      "tol_state": tol_state}))
        return False

    def scan(self) -> bool:
        """One detection scan.  Returns True if a completion was accepted."""
        self.n_scans += 1
        active_keys = set()
        thr_split = self.cfg.d_split * self.cfg.d_watch * self.kern.sigma
        for k in range(len(self.blocks) - 1, -1, -1):
            b = self.blocks[k]
            if (not b.confluent) and b.diameter > thr_split:
                self.split(k)
        # regroup groups of several blocks inside the watch radius (exact change of coordinates)
        for _ in range(len(self.blocks)):
            done = True
            for members, extent in sorted(self.groups(), key=lambda g: -len(g[0])):
                if len(members) >= 2:
                    if self.regroup(members):
                        done = False
                        break
            if done:
                break
        accepted = False
        for members, extent in self.groups():
            blocks = [self.blocks[i] for i in members]
            kappa_C = self.diagnostics(blocks)
            flag = kappa_C <= self.cfg.kappa_on and extent <= self.cfg.d_on * self.kern.sigma
            gkey = (round(float(np.mean([b.beta for b in blocks])), 1), len(members), tuple(int(b.nu) for b in blocks))
            active_keys.add(gkey)
            self.flag_counters[gkey] = self.flag_counters.get(gkey, 0) + 1 if flag else 0
            self.log(f"  scan {self.n_scans}: group {members} extent={extent:.3e} kappa={kappa_C:.2e} flag={flag} count={self.flag_counters[gkey]}")
            if self.cfg.completion and not accepted and self.flag_counters[gkey] >= self.cfg.persistence:
                if len(members) == 1:
                    if self.attempt_completion(members[0]):
                        accepted = True
                        break
                else:
                    self.unresolved_flags += 1
        for k in list(self.flag_counters):
            if k not in active_keys:
                del self.flag_counters[k]
        return accepted

    # ---- residual search and unconstrained insertion -------------------------
    def residual_search(self, search_candidates: bool = True):
        """Summarize the error, optionally searching insertion candidates; no global certificate."""
        at = self.atoms()
        cands = []
        orders = range(min(self.cfg.r_max_insert, self.cfg.N-self.slots-1)+1) if search_candidates else ()
        if orders:
            resid = self.prob.residual_atoms(at)
            grid = np.linspace(-self.cfg.grid_L, self.cfg.grid_L, self.cfg.grid_n)
            hgrid = grid[1] - grid[0]
        for r in orders:
            qgrid = self.prob.correlations(resid, r, grid)
            i = int(np.argmax(np.abs(qgrid)))
            lo, hi = max(-self.cfg.grid_L, grid[i]-hgrid), min(self.cfg.grid_L, grid[i]+hgrid)
            objective = lambda b, r=r: -abs(float(self.prob.correlations(resid,r,np.array([b]))[0]))
            result = minimize_scalar(objective, bounds=(lo,hi), method="bounded", options={"xatol":1e-12})
            center = float(result.x) if result.success else float(grid[i])
            q = float(self.prob.correlations(resid,r,np.array([center]))[0])
            cands.append((q*q/(2*self.kern.kappa(r)**2),r,center,q))
        cands.sort(reverse=True)
        result = dict(candidates=cands,rel_err=self.prob.rel_error(at),excess=self.prob.excess(at),
                      global_certificate=False)
        self.last_cert = result
        return result

    def insertion_trial(self, cand):
        """Append alpha G^(r), alpha=q/||G^(r)||², then refit all coefficients."""
        _, r, center, q = cand
        if self.slots+r+1>self.cfg.N or q==0:
            return None
        alpha=q/self.kern.kappa(r)**2
        before=self.prob.excess(self.atoms())
        blocks_new=[block.copy() for block in self.blocks]
        gamma=np.zeros(r+1); gamma[r]=(-1.)**r*alpha
        blocks_new.append(Block([center],gamma,[r+1]))
        step=self.prob.excess(self.atoms(blocks_new))
        fitted,*_=self.fit(blocks_new,warm=None)
        self.set_gamma(blocks_new,fitted)
        after=self.prob.excess(self.atoms(blocks_new))
        info=dict(order=r,center=center,alpha=alpha,t=alpha,a=q,dJ_step=step-before,
                  dJ=after-before,line_model_decrease=q*q/(2*self.kern.kappa(r)**2))
        return blocks_new,info

    def insert(self, cands) -> bool:
        """Choose among the best candidate of each order and apply the insertion."""
        trials = []
        for cand in cands:
            res = self.insertion_trial(cand)
            if res is not None:
                trials.append(res)
                if self.cfg.insert_rule == "correlation":
                    break           # candidates are sorted by their unconstrained line decrease
        if not trials:
            return False
        if self.cfg.insert_rule == "correlation":
            blocks_new, info = trials[0]
        elif self.cfg.insert_rule == "decrease":
            blocks_new, info = min(trials, key=lambda tr: tr[1]["dJ"])
        else:  # "per_slot": largest measured decrease per slot used
            blocks_new, info = min(trials, key=lambda tr: tr[1]["dJ"] / (tr[1]["order"] + 1))
        J_now = self.prob.J(self.atoms())
        if info["dJ"] >= -self.cfg.insert_min_decrease * max(1.0, abs(J_now)):
            self.trace.events.append((self.n_fits, "insertion_negligible", info))
            return False
        self.blocks = blocks_new
        self.gamma_prev = None
        self.n_insertions += 1
        info["slots"] = self.slots
        info["n_trials"] = len(trials)
        self.trace.events.append((self.n_fits, "insertion", info))
        self.log(f"  insertion: order {info['order']} at b={info['center']:.4f}, t={info['t']:.3e}, a={info['a']:.3e}, dJ step {info['dJ_step']:+.3e}, after refit {info['dJ']:+.3e}, slots={self.slots} (of {len(trials)} trials)")
        self.record()
        return True

    def prune(self) -> bool:
        """Controlled removal of vanishing trailing orders (frees slots)."""
        if not self.cfg.prune:
            return False
        at = self.atoms()
        v_norm = math.sqrt(max(self.kern.norm2(at), 0.0))
        B_v = self.prob.error_norm(at)
        tol = self.cfg.prune_rel * v_norm
        changed = False
        # whole negligible blocks (T_C = 0)
        for k in range(len(self.blocks) - 1, -1, -1):
            if len(self.blocks) <= 1:
                break
            b = self.blocks[k]
            vb = b.atoms(self.kern)
            e = math.sqrt(max(self.kern.norm2(vb), 0.0)) * (1 + 1e-10)
            eps_j = self.next_allowance()
            if e <= tol and B_v * e + 0.5 * e * e <= eps_j:
                J_before = self.prob.J(self.atoms())
                self.blocks.pop(k)
                self.j_allow += 1
                gamma, *_ = self.fit(self.blocks, warm=None)
                self.set_gamma(self.blocks, gamma)
                J_after = self.prob.J(self.atoms())
                self.trace.events.append((self.n_fits, "delete", {"beta": b.beta, "nu": b.nu, "e": e, "eps_j": eps_j, "dJ": J_after - J_before}))
                self.log(f"  delete block at {b.beta:.4f} (nu {b.nu}): e={e:.2e}, dJ={J_after-J_before:+.2e}")
                changed = True
        for k, b in enumerate(self.blocks):
            if not b.confluent or b.nu < 2:
                continue
            best = None
            for s0 in range(1, b.nu):
                tail = AtomTable(np.full(b.nu - s0, b.beta), np.arange(s0, b.nu),
                                 np.array([((-1.0) ** s) * b.gamma[s] for s in range(s0, b.nu)]))
                e = math.sqrt(max(self.kern.norm2(tail), 0.0)) * (1 + 1e-10)
                eps_j = self.next_allowance()
                if e <= tol and B_v * e + 0.5 * e * e <= eps_j:
                    best = (s0, e, eps_j)
                    break
            if best is not None:
                s0, e, eps_j = best
                J_before = self.prob.J(self.atoms())
                self.blocks[k] = Block(b.nodes.copy(), b.gamma[:s0].copy(), [s0])
                self.j_allow += 1
                gamma, *_ = self.fit(self.blocks, warm=None)
                self.set_gamma(self.blocks, gamma)
                J_after = self.prob.J(self.atoms())
                self.trace.events.append((self.n_fits, "prune", {"beta": b.beta, "nu_old": b.nu, "nu_new": s0, "e": e,
                                                                  "eps_j": eps_j, "dJ": J_after - J_before}))
                self.log(f"  prune block at {b.beta:.4f}: nu {b.nu} -> {s0}, e={e:.2e}, dJ={J_after-J_before:+.2e}")
                changed = True
        if changed:
            self.gamma_prev = None
            self.record()
        return changed

    # ---- main loop ------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        t0 = time.time()
        gamma, *_ = self.fit(self.blocks, warm=None)
        self.set_gamma(self.blocks, gamma)
        self.gamma_prev = gamma
        self.record()
        self.log(f"start: rel err {self.trace.rel_err[-1]:.3e}, coefficient cost {self.cost():.4f} (unconstrained), slots {self.slots}")
        for _ in range(len(self.blocks)):
            if not self.scan():
                break
        stalls = 0
        pending_rounds = 0
        while self.n_fits < cfg.max_fits and self.lbfgs_iters < cfg.lbfgs_maxiter_total:
            res = self.optimize_centers(cfg.scan_period)
            converged = res.nit < cfg.scan_period or res.status != 1
            if self.scan():
                stalls = 0
                continue
            if not converged:
                continue
            cert = self.residual_search(search_candidates=cfg.insertion)
            self.log(f"  optimizer return ({res.message}): rel err {cert['rel_err']:.3e}, slots {self.slots}/{cfg.N}")
            inserted = False
            if cfg.insertion:
                if self.prune():
                    cert = self.residual_search()
                inserted = self.insert(cert["candidates"])
            if inserted:
                stalls = 0
                pending_rounds = 0
                continue
            if any(v > 0 for v in self.flag_counters.values()) and pending_rounds < cfg.persistence + 2:
                pending_rounds += 1          # a flagged group awaits its persistence check
                continue
            stalls += 1
            if stalls >= 2:
                self.stop_reason = "stationary"
                break
        if self.stop_reason is None:
            self.stop_reason = "work_limit"
        self.normal_return = (self.abnormal_returns == 0)
        cert = self.residual_search(search_candidates=cfg.insertion)
        self.wall = time.time() - t0
        return cert


# ----------------------------------------------------------------------------
# Standard optimization baseline: ordinary Gaussians, unconstrained least squares
# ----------------------------------------------------------------------------

class StandardSolver1D:
    def __init__(self, prob: Problem1D, centers: np.ndarray, max_fits: int = 3000, scan_period: int = 20,
                 cond_limit: float = 1e16, lbfgs_ftol: float = 1e-15, lbfgs_gtol: float = 1e-12, verbose: bool = False):
        self.prob = prob
        self.kern = prob.kern
        self.centers = np.asarray(centers, dtype=float).copy()
        self.P = len(self.centers)
        self.max_fits = max_fits
        self.scan_period = scan_period
        self.cond_limit = cond_limit
        self.ftol, self.gtol = lbfgs_ftol, lbfgs_gtol
        self.n_fits = 0
        self.trace = Trace()
        self.weights = np.zeros(self.P)
        self.stop_reason = None
        self.normal_return = True
        self.verbose = verbose
        self.best_rel = np.inf
        self.lbfgs_iters = 0
        self.max_cond = 0.0

    def atoms(self, centers=None, weights=None):
        c = self.centers if centers is None else centers
        w = self.weights if weights is None else weights
        return AtomTable(np.asarray(c, float), np.zeros(len(c), dtype=int), np.asarray(w, float))

    def fit(self, centers, strict: bool = True):
        """Unconstrained least squares at fixed centers.  At trial points (strict=False) an
        ill-conditioned system is directly solved without rank truncation so that the line search
        can back off; an accepted iterate above the condition limit stops the run."""
        A = AtomTable(np.asarray(centers, float), np.zeros(len(centers), dtype=int), np.ones(len(centers)))
        M = self.kern.gram(A)
        g = self.kern.gram(A, self.prob.target) @ self.prob.target.coeffs
        d = np.sqrt(np.diag(M))
        Mn = M / d[:, None] / d[None, :]
        self.n_fits += 1
        cond = np.linalg.cond(Mn)
        if strict:
            self.max_cond = max(self.max_cond, cond if np.isfinite(cond) else np.inf)
        if not np.isfinite(cond) or cond > self.cond_limit:
            if strict:
                return None, M, g, cond
            try:
                w, _ = solve_unconstrained_qp(M, g)
            except (np.linalg.LinAlgError, FloatingPointError):
                return None, M, g, cond
            return w, M, g, cond
        try:
            w, _ = solve_unconstrained_qp(M, g)
        except (np.linalg.LinAlgError, FloatingPointError):
            return None, M, g, cond
        return w, M, g, cond

    def objective(self, p):
        w, M, g, cond = self.fit(p, strict=False)
        if w is None or not np.all(np.isfinite(w)):
            raise FloatingPointError("ill-conditioned coefficient system")
        J = 0.5 * w @ M @ w - g @ w
        at = self.atoms(p, w)
        resid = self.prob.residual_atoms(at)      # u* - v
        q1 = self.prob.correlations(resid, 1, p)  # (u* - v, G'_{b_i})_H
        grad = w * q1                              # (v - u*, -w_i G'_{b_i})_H
        return J, grad

    def record(self):
        at = self.atoms()
        rel = self.prob.rel_error(at)
        self.best_rel = min(self.best_rel, rel)
        tr = self.trace
        tr.fits.append(self.n_fits); tr.rel_err.append(rel); tr.rel_err_quad.append(self.prob.rel_error_quad(at)); tr.energy.append(self.prob.J(at))
        tr.coef_norm.append(float(np.linalg.norm(self.weights))); tr.raw_norm.append(float(np.linalg.norm(self.weights)))
        c = np.sort(self.centers)
        tr.min_halfgap.append(float(np.diff(c).min() / 2) if len(c) > 1 else np.inf)
        tr.cost.append(np.nan); tr.slots.append(self.P)

    def run(self):
        t0 = time.time()
        w, M, g, cond = self.fit(self.centers)
        if w is None:
            self.stop_reason = "ill-conditioned at start"
            self.normal_return = False
            self.record()
            self.wall = time.time() - t0
            return
        self.weights = w
        self.record()
        bounds = [(-self.prob.L0, self.prob.L0)] * self.P
        self._nrec = 0

        def cb(xk):
            self._nrec += 1
            if self._nrec % self.scan_period == 0:
                w, M, g, cond = self.fit(xk, strict=False)
                if w is not None and np.all(np.isfinite(w)):
                    self.centers, self.weights = xk.copy(), w
                    self.record()
            if self.n_fits >= self.max_fits:
                raise StopIteration
        while self.n_fits < self.max_fits:
            try:
                res = minimize(self.objective, self.centers, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                               options={"maxiter": 100000, "ftol": self.ftol, "gtol": self.gtol, "maxls": 40})
            except StopIteration:
                self.stop_reason = "work_limit"
                w, M, g, cond = self.fit(self.centers)
                if w is not None:
                    self.weights = w
                self.record()
                break
            except FloatingPointError:
                self.stop_reason = "ill-conditioned"
                self.normal_return = False
                break
            self.lbfgs_iters += res.nit
            w, M, g, cond = self.fit(res.x)
            if w is None:
                self.stop_reason = "ill-conditioned"
                self.normal_return = False
                break
            self.centers, self.weights = res.x.copy(), w
            self.record()
            if res.status == 0:
                self.stop_reason = "converged"
                break
            if res.status == 2:
                self.stop_reason = "abnormal: " + str(res.message)
                self.normal_return = False
                break
        if self.stop_reason is None:
            self.stop_reason = "work_limit"
        self.wall = time.time() - t0
