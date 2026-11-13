"""Analytic H^1(R) calculus for translated Gaussian derivatives.

G(x) = exp(-x^2/(2 sigma^2)),  H = H^1(R) with (v,w)_H = int (v w + v' w').

Correlation kernel:  Gamma(z) = int G(x) G(x-z) dx = sqrt(pi) sigma exp(-z^2/(4 sigma^2)),
    (G^{(r)}(.-b), G^{(s)}(.-c))_{L^2} = (-1)^r Gamma^{(r+s)}(b-c),
    (G^{(r)}(.-b), G^{(s)}(.-c))_{H}   = (-1)^r K^{(r+s)}(b-c),   K = Gamma - Gamma''.
Norms:  kappa_r^2 = (-1)^r K^{(2r)}(0).

Newton (divided-difference) blocks are expanded in confluent derivative atoms at the
block centroid through the Hermite--Genocchi moment identity
    Phi_r(t_0..t_r) = (-1)^r sum_k r!/(k+r)! h_k(beta-t_0,...,beta-t_r) G^{(r+k)}(.-beta),
where h_k is the complete homogeneous symmetric polynomial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.special import eval_hermite, eval_hermitenorm, gammaln


# ----------------------------------------------------------------------------
# Kernel derivatives
# ----------------------------------------------------------------------------

class Kernel1D:
    """Gaussian of width sigma with its analytic H^1 Gram calculus."""

    def __init__(self, sigma: float = 1.0, max_order: int = 80):
        self.sigma = float(sigma)
        self.max_order = int(max_order)
        self._kappa = None

    # Gamma^{(n)}(z) = sqrt(pi) sigma (-1)^n (2 sigma)^{-n} H_n(z/(2 sigma)) exp(-z^2/(4 sigma^2))
    def Gamma_n(self, n: int, z):
        z = np.asarray(z, dtype=float)
        s = self.sigma
        x = z / (2.0 * s)
        return math.sqrt(math.pi) * s * ((-1.0) ** n) * (2.0 * s) ** (-n) * eval_hermite(n, x) * np.exp(-x * x)

    def K_n(self, n: int, z):
        """n-th derivative of the H^1 correlation kernel K = Gamma - Gamma''."""
        return self.Gamma_n(n, z) - self.Gamma_n(n + 2, z)

    def K_n_envelope(self, n: int, D: float) -> float:
        """Rigorous (up to rounding) bound for sup_{|z|>=D} |K^{(n)}(z)|.

        Writes K^{(n)}(z) = exp(-z^2/(4 sigma^2)) * sum_j c_{n,j} z^j and bounds each monomial
        times the Gaussian by its maximum on [D, inf), attained at a_j = max(D, sigma sqrt(2 j)).
        """
        s = self.sigma
        coeffs = _K_poly_coeffs(n, s)  # polynomial coefficients c_{n,j}
        total = 0.0
        for j, c in enumerate(coeffs):
            if c == 0.0:
                continue
            a = max(D, s * math.sqrt(2.0 * j))
            total += abs(c) * (a ** j) * math.exp(-a * a / (4.0 * s * s))
        return total

    def Gamma_envelope(self, n: int, D: float) -> float:
        """Upper bound for sup_{|z|>=D} |Gamma^{(n)}(z)| (D = 0 gives a global bound)."""
        s = self.sigma
        coeffs = _Gamma_poly_coeffs(n, s)
        total = 0.0
        for j, c in enumerate(coeffs):
            if c == 0.0:
                continue
            a = max(D, s * math.sqrt(2.0 * j))
            total += abs(c) * (a ** j) * math.exp(-a * a / (4.0 * s * s))
        return total

    def kappa(self, r: int) -> float:
        """kappa_r = ||G^{(r)}||_H."""
        if self._kappa is None:
            self._kappa = {}
        if r not in self._kappa:
            val = ((-1.0) ** r) * float(self.K_n(2 * r, 0.0))
            self._kappa[r] = math.sqrt(max(val, 0.0))
        return self._kappa[r]

    def kappas(self, rmax: int) -> np.ndarray:
        return np.array([self.kappa(r) for r in range(rmax + 1)])

    # pointwise values of G^{(r)}(x)
    def G_r(self, r: int, x):
        x = np.asarray(x, dtype=float)
        s = self.sigma
        y = x / s
        return ((-1.0) ** r) * s ** (-r) * eval_hermitenorm(r, y) * np.exp(-0.5 * y * y)

    # Gram between atom tables ------------------------------------------------
    def gram(self, A: "AtomTable", B: "AtomTable" | None = None) -> np.ndarray:
        """Matrix of (atom_i of A, atom_j of B)_H."""
        if B is None:
            B = A
        Z = A.centers[:, None] - B.centers[None, :]
        N = A.orders[:, None] + B.orders[None, :]
        sign = ((-1.0) ** A.orders)[:, None]
        out = np.empty(Z.shape)
        for n in np.unique(N):
            mask = N == n
            out[mask] = self.K_n(int(n), Z[mask])
        return sign * out

    def norm2(self, A: "AtomTable") -> float:
        M = self.gram(A)
        return float(A.coeffs @ M @ A.coeffs)

    def inner(self, A: "AtomTable", B: "AtomTable") -> float:
        return float(A.coeffs @ self.gram(A, B) @ B.coeffs)


@lru_cache(maxsize=None)
def _Gamma_poly_coeffs(n: int, sigma: float) -> tuple:
    """Polynomial coefficients of Gamma^{(n)}(z) = exp(-z^2/(4 s^2)) sum_j c_j z^j."""
    from numpy.polynomial import hermite as Hm
    s = sigma
    c = np.zeros(n + 1); c[n] = 1.0
    p = Hm.herm2poly(c) * (1.0 / (2.0 * s)) ** np.arange(n + 1)
    p = math.sqrt(math.pi) * s * ((-1.0) ** n) * (2.0 * s) ** (-n) * p
    return tuple(float(x) for x in p)


@lru_cache(maxsize=None)
def _K_poly_coeffs(n: int, sigma: float) -> tuple:
    """Polynomial coefficients c_{n,j} of K^{(n)}(z) = exp(-z^2/(4 s^2)) sum_j c_{n,j} z^j.

    Uses Gamma^{(n)}(z) = sqrt(pi) s (-1)^n (2s)^{-n} H_n(z/(2s)) exp(-z^2/(4s^2)).
    """
    from numpy.polynomial import hermite as Hm
    s = sigma

    def gamma_poly(m):
        # H_m(z/(2s)) as polynomial in z
        c = np.zeros(m + 1)
        c[m] = 1.0
        p = Hm.herm2poly(c)  # coefficients in x = z/(2s)
        p = p * (1.0 / (2.0 * s)) ** np.arange(m + 1)
        return math.sqrt(math.pi) * s * ((-1.0) ** m) * (2.0 * s) ** (-m) * p

    g1 = gamma_poly(n)
    g2 = gamma_poly(n + 2)
    out = np.zeros(n + 3)
    out[: n + 1] += g1
    out[: n + 3] -= g2
    return tuple(float(x) for x in out)


# ----------------------------------------------------------------------------
# Atom tables: lists of (center, order, coefficient)
# ----------------------------------------------------------------------------

@dataclass
class AtomTable:
    centers: np.ndarray
    orders: np.ndarray
    coeffs: np.ndarray

    @staticmethod
    def empty():
        return AtomTable(np.zeros(0), np.zeros(0, dtype=int), np.zeros(0))

    @staticmethod
    def concat(tables):
        tables = [t for t in tables if t.centers.size]
        if not tables:
            return AtomTable.empty()
        return AtomTable(np.concatenate([t.centers for t in tables]),
                         np.concatenate([t.orders for t in tables]).astype(int),
                         np.concatenate([t.coeffs for t in tables]))

    def scaled(self, factor: float):
        return AtomTable(self.centers.copy(), self.orders.copy(), factor * self.coeffs)

    def minus(self, other: "AtomTable"):
        return AtomTable.concat([self, other.scaled(-1.0)])

    def evaluate(self, kern: Kernel1D, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for c, r, a in zip(self.centers, self.orders, self.coeffs):
            if a != 0.0:
                out += a * kern.G_r(int(r), x - c)
        return out

    def __len__(self):
        return len(self.centers)


# ----------------------------------------------------------------------------
# Complete homogeneous symmetric polynomials and Newton-block expansions
# ----------------------------------------------------------------------------

def complete_homogeneous(x: np.ndarray, kmax: int) -> np.ndarray:
    """h_k(x_0..x_n) for k = 0..kmax via the recurrence h_k(x_0..x_j) = h_k(x_0..x_{j-1}) + x_j h_{k-1}(x_0..x_j)."""
    h = np.zeros(kmax + 1)
    h[0] = 1.0
    for xj in x:
        for k in range(1, kmax + 1):
            h[k] = h[k] + xj * h[k - 1]
    return h


def expansion_matrix(nodes: np.ndarray, beta: float, K_extra: int) -> np.ndarray:
    """Matrix T (S x nu) with Phi_r(t_0..t_r) = sum_s T[s,r] G^{(s)}(.-beta), S = nu + K_extra.

    T[s,r] = (-1)^r r!/s! h_{s-r}(beta - t_0, ..., beta - t_r)  for s >= r.
    """
    nodes = np.asarray(nodes, dtype=float)
    nu = len(nodes)
    S = nu + K_extra
    tau = beta - nodes
    T = np.zeros((S, nu))
    for r in range(nu):
        h = complete_homogeneous(tau[: r + 1], S - 1 - r)
        s = np.arange(r, S)
        # r!/s! = exp(gammaln(r+1) - gammaln(s+1))
        fac = np.exp(gammaln(r + 1) - gammaln(s + 1))
        T[s, r] = ((-1.0) ** r) * fac * h
    return T


def expansion_matrix_derivative(nodes: np.ndarray, beta: float, K_extra: int, j: int) -> np.ndarray:
    """d T / d t_j at fixed expansion point beta.

    d/dt_j h_k(tau_0..tau_r) = - h_{k-1}(tau_0..tau_r, tau_j)   (tau = beta - t), for j <= r.
    """
    nodes = np.asarray(nodes, dtype=float)
    nu = len(nodes)
    S = nu + K_extra
    tau = beta - nodes
    dT = np.zeros((S, nu))
    for r in range(j, nu):
        ext = np.concatenate([tau[: r + 1], [tau[j]]])
        h = complete_homogeneous(ext, S - 1 - r)  # h_0..h_{S-1-r}
        s = np.arange(r + 1, S)  # k = s - r >= 1
        fac = np.exp(gammaln(r + 1) - gammaln(s + 1))
        dT[s, r] = -((-1.0) ** r) * fac * h[s - r - 1]
    return dT


def needed_extra_order(kern: Kernel1D, delta: float, nu: int, rel_tol: float = 1e-17, K_min: int = 6, K_max: int = 60) -> int:
    """Smallest K such that the Taylor tail bound sum_{k>K} delta^k/k! kappa_{r+k}/kappa_r is below rel_tol for all r<nu."""
    if delta <= 0.0:
        return K_min
    for K in range(K_min, K_max + 1):
        ok = True
        for r in range(nu):
            # bound the tail by its first term times a geometric factor 2 (terms decrease fast)
            k = K + 1
            term = (delta ** k) / math.factorial(k) * kern.kappa(r + k) / kern.kappa(r)
            if 2.0 * term > rel_tol:
                ok = False
                break
        if ok:
            return K
    return K_max


# ----------------------------------------------------------------------------
# Shifting expansions between centers
# ----------------------------------------------------------------------------

def shift_atoms(kern: Kernel1D, table: AtomTable, beta_new: float, rel_tol: float = 1e-17) -> AtomTable:
    """Re-expand every atom of a table at the common center beta_new (exact Taylor shift,
    truncated with a certified relative tail below rel_tol).  G^{(s)}(.-c) = sum_k (beta-c)^k/k! G^{(s+k)}(.-beta)."""
    if len(table) == 0:
        return AtomTable(np.zeros(0), np.zeros(0, dtype=int), np.zeros(0))
    smax = int(table.orders.max())
    delta = float(np.max(np.abs(beta_new - table.centers)))
    K = needed_extra_order(kern, delta, smax + 1, rel_tol=rel_tol)
    S = smax + 1 + K
    c = np.zeros(S)
    for cen, s, a in zip(table.centers, table.orders, table.coeffs):
        if a == 0.0:
            continue
        d = beta_new - cen
        k = np.arange(0, S - s)
        c[s:] += a * (d ** k) / np.exp(gammaln(k + 1))
    return AtomTable(np.full(S, beta_new), np.arange(S), c)


# ----------------------------------------------------------------------------
# Newton blocks with node multiplicities
# ----------------------------------------------------------------------------

@dataclass
class Block:
    """A Newton divided-difference block with distinct nodes and multiplicities.

    nodes: distinct node values (sorted);  mult: multiplicities;  gamma: nu = sum(mult) coefficients.
    The repeated node list t_0 <= ... <= t_{nu-1} defines the nested Newton columns
        Phi_r = r![t_0..t_r] F,   F(t) = G(.-t),
    and  v_block = sum_r gamma_r Phi_r.  A single distinct node (confluent block) gives
        v_block = sum_r (-1)^r gamma_r G^{(r)}(.-beta).
    """
    nodes: np.ndarray
    gamma: np.ndarray
    mult: np.ndarray | None = None

    def __post_init__(self):
        self.nodes = np.atleast_1d(np.asarray(self.nodes, dtype=float)).copy()
        self.gamma = np.atleast_1d(np.asarray(self.gamma, dtype=float)).copy()
        if self.mult is None:
            self.mult = np.ones(len(self.nodes), dtype=int)
        self.mult = np.atleast_1d(np.asarray(self.mult, dtype=int)).copy()
        assert len(self.mult) == len(self.nodes)
        assert int(self.mult.sum()) == len(self.gamma), (self.mult, self.gamma)
        # NOTE: the node order is part of the nested Newton basis and is never re-sorted here.

    @property
    def nu(self) -> int:
        return int(self.mult.sum())

    @property
    def confluent(self) -> bool:
        return len(self.nodes) == 1

    @property
    def beta(self) -> float:
        return float(np.sum(self.nodes * self.mult) / self.mult.sum())

    @property
    def diameter(self) -> float:
        return float(self.nodes.max() - self.nodes.min())

    def all_nodes(self) -> np.ndarray:
        return np.repeat(self.nodes, self.mult)

    def copy(self):
        return Block(self.nodes.copy(), self.gamma.copy(), self.mult.copy())

    def expansion(self, kern: Kernel1D, K_min_extra: int = 1):
        """Return (beta, T) with the block's columns expanded in G^{(s)}(.-beta), s < nu + K."""
        beta = self.beta
        if self.confluent:
            nu = self.nu
            T = np.zeros((nu + K_min_extra, nu))
            for r in range(nu):
                T[r, r] = (-1.0) ** r
            return beta, T
        tn = self.all_nodes()
        delta = float(np.max(np.abs(beta - tn)))
        K = max(needed_extra_order(kern, delta, self.nu), K_min_extra)
        return beta, expansion_matrix(tn, beta, K)

    def atoms(self, kern: Kernel1D) -> AtomTable:
        beta, T = self.expansion(kern)
        c = T @ self.gamma
        S = T.shape[0]
        return AtomTable(np.full(S, beta), np.arange(S), c)

    def cost(self, rho: np.ndarray) -> float:
        return float(np.sum(rho[: self.nu] * np.abs(self.gamma)))


def raw_weights_of_block(block: "Block") -> np.ndarray:
    """Ordinary-coordinate weights w_j of a block with simple nodes (all multiplicities one):
    sum_r gamma_r Phi_r(t_0..t_r) = sum_j w_j G(.-t_j),  w_j = sum_{r>=j} gamma_r r! / prod_{k<=r, k!=j} (t_j - t_k)."""
    assert np.all(block.mult == 1)
    t = block.nodes
    nu = block.nu
    w = np.zeros(nu)
    for r in range(nu):
        if block.gamma[r] == 0.0:
            continue
        for j in range(r + 1):
            denom = 1.0
            for k in range(r + 1):
                if k != j:
                    denom *= (t[j] - t[k])
            w[j] += block.gamma[r] * math.factorial(r) / denom
    return w


def newton_union(kern: Kernel1D, members: list) -> "Block":
    """Exact regrouping of several blocks into one Newton block on the union of their nodes
    (multiplicities added for coincident nodes).

    The coefficients are obtained by matching the Taylor expansions at the union centroid:
    the expansion matrix of the union columns is lower triangular with unit-modulus diagonal,
    so the first nu orders determine gamma exactly (triangular solve, no Gram system).
    """
    members = list(members)
    # exactly coincident confluent blocks span the same derivative directions: add coefficients
    merged = []
    for b in members:
        if b.confluent:
            hit = next((m for m in merged if m.confluent and m.nodes[0] == b.nodes[0]), None)
            if hit is not None:
                nu = max(hit.nu, b.nu)
                g = np.zeros(nu); g[: hit.nu] += hit.gamma; g[: b.nu] += b.gamma
                merged[merged.index(hit)] = Block(hit.nodes.copy(), g, [nu])
                continue
        merged.append(b)
    members = merged
    if len(members) == 1 and members[0].confluent:
        return members[0].copy()
    nodes = np.concatenate([b.nodes for b in members])
    mult = np.concatenate([b.mult for b in members])
    # merge coincident nodes
    order = np.argsort(nodes, kind="stable")
    nodes, mult = nodes[order], mult[order]
    un, um = [], []
    for t, m in zip(nodes, mult):
        if un and t == un[-1]:
            um[-1] += int(m)
        else:
            un.append(float(t)); um.append(int(m))
    nodes, mult = np.array(un), np.array(um, dtype=int)
    nu = int(mult.sum())
    trial = Block(nodes, np.zeros(nu), mult)
    beta, T = trial.expansion(kern)
    tables = [shift_atoms(kern, b.atoms(kern), beta) for b in members]
    S = max(len(t) for t in tables)
    c = np.zeros(S)
    for t in tables:
        c[: len(t)] += t.coeffs
    Tn = T[:nu, :nu]
    gam = np.linalg.solve(Tn, c[:nu])
    return Block(nodes, gam, mult)


def newton_from_raw(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact Newton coefficients of sum_j w_j G(.-t_j) in the basis Phi_r(t_0..t_r) (simple nodes):
    gamma_r = (1/r!) sum_{j>=r} w_j prod_{k<r} (t_j - t_k)."""
    t = np.asarray(nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    nu = len(t)
    gam = np.zeros(nu)
    for r in range(nu):
        acc = 0.0
        for j in range(r, nu):
            prod = 1.0
            for k in range(r):
                prod *= (t[j] - t[k])
            acc += w[j] * prod
        gam[r] = acc / math.factorial(r)
    return gam
