"""Closed-form H^1(R^d) inner products and point values of directional Gaussian derivatives.

A primitive is (c, u, a, w): w * d_u^a G(. - c), with G(x) = exp(-|x|^2/(2 sigma^2)), |u| = 1, a <= 4.
With k(z) = (G(.-x), G(.-y))_{L2} = (pi sigma^2)^{d/2} exp(-|z|^2/(4 sigma^2)), z = x - y, and
K = (1 - Laplacian) k, one has (d_u^a G(.-x), d_v^b G(.-y))_H = (-1)^a d_u^a d_v^b K(z), and
sum_{a,b} t^a s^b/(a! b!) d_u^a d_v^b K(z) = K(z + t u + s v).  The Taylor coefficients of
k(z + t u + s v) are Gaussian times two-direction Hermite polynomials,
  [t^a s^b] k = C exp(-|z|^2/(2 tau^2)) tau^(-a-b) sum_k (-rho)^k He_{a-k}(-u.z/tau) He_{b-k}(-v.z/tau)
                / ((a-k)! (b-k)! k!),            tau^2 = 2 sigma^2, rho = u.v,
and K(w) = (1 + d/tau^2 - |w|^2/tau^4) k(w) gives the coefficients of K.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import torch

MAXORD = 9          # Phase 1 (11 Nov 2026): orders 5..9 added for t-pair series and opening slopes
FACT = [math.factorial(i) for i in range(2 * MAXORD + 2)]


def _he(n, y):
    """Probabilists' Hermite polynomial He_n(y) (works for torch tensors and mpmath numbers).

    Orders n <= 4 use the original explicit formulas, so all earlier computations are reproduced
    bit for bit; orders 5 <= n <= MAXORD use He_{k+1} = y He_k - k He_{k-1}."""
    if n == 0:
        return y * 0 + 1
    if n == 1:
        return y
    if n == 2:
        return y * y - 1
    if n == 3:
        return y ** 3 - 3 * y
    if n == 4:
        return y ** 4 - 6 * y * y + 3
    if n <= MAXORD:
        prev, cur = y ** 3 - 3 * y, y ** 4 - 6 * y * y + 3
        for k in range(4, n):
            prev, cur = cur, y * cur - k * prev
        return cur
    raise ValueError(n)


def _k_coeffs(alpha, beta, rho, r2, tau, C, amax, bmax):
    """Table c[a][b] = [t^a s^b] k(z + t u + s v) for a <= amax, b <= bmax."""
    E = C * (torch.exp(-r2 / (2 * tau * tau)) if torch.is_tensor(r2) else mp.e ** (-r2 / (2 * tau * tau)))
    ya, yb = -alpha / tau, -beta / tau
    Ha = [_he(n, ya) for n in range(amax + 1)]
    Hb = [_he(n, yb) for n in range(bmax + 1)]
    c = [[None] * (bmax + 1) for _ in range(amax + 1)]
    for a in range(amax + 1):
        for b in range(bmax + 1):
            acc = 0
            for k in range(min(a, b) + 1):
                acc = acc + ((-rho) ** k) * Ha[a - k] * Hb[b - k] / (FACT[a - k] * FACT[b - k] * FACT[k])
            c[a][b] = E * acc / tau ** (a + b)
    return c


def _K_coeff(c, a, b, alpha, beta, rho, P0, tau):
    get = lambda i, j: c[i][j] if (i >= 0 and j >= 0) else 0
    return (P0 * get(a, b)
            - (2 * alpha * get(a - 1, b) + 2 * beta * get(a, b - 1) + get(a - 2, b)
               + 2 * rho * get(a - 1, b - 1) + get(a, b - 2)) / tau ** 4)


def h1_gram(cA, uA, oA, cB, uB, oB, sigma):
    """Matrix of (d_{uA_i}^{oA_i} G(.-cA_i), d_{uB_j}^{oB_j} G(.-cB_j))_{H^1} (torch, differentiable)."""
    d = cA.shape[1]
    tau = math.sqrt(2.0) * sigma
    C = (math.pi * sigma * sigma) ** (d / 2)
    z = cA[:, None, :] - cB[None, :, :]
    r2 = (z * z).sum(-1)
    alpha = (uA[:, None, :] * z).sum(-1)
    beta = (uB[None, :, :] * z).sum(-1)
    rho = uA @ uB.T
    amax, bmax = int(oA.max()), int(oB.max())
    c = _k_coeffs(alpha, beta, rho, r2, tau, C, amax + 0, bmax + 0)
    P0 = 1 + d / tau ** 2 - r2 / tau ** 4
    table = torch.stack([torch.stack([_K_coeff(c, a, b, alpha, beta, rho, P0, tau) for b in range(bmax + 1)])
                         for a in range(amax + 1)])                     # (amax+1, bmax+1, n, m)
    ia = torch.as_tensor(oA, dtype=torch.long)[:, None]
    ib = torch.as_tensor(oB, dtype=torch.long)[None, :]
    n, m = cA.shape[0], cB.shape[0]
    K = table[ia, ib, torch.arange(n)[:, None], torch.arange(m)[None, :]]
    sign_fact = torch.tensor([[(-1) ** int(a) * FACT[int(a)] * FACT[int(b)] for b in oB] for a in oA], dtype=K.dtype)
    return sign_fact * K


def point_values(X, c, u, o, sigma):
    """Matrix of d_{u_j}^{o_j} G(X_i - c_j) (torch, differentiable); X has shape (N, d)."""
    X2 = (X * X).sum(1)[:, None]
    c2 = (c * c).sum(1)[None, :]
    r2 = torch.clamp(X2 + c2 - 2 * X @ c.T, min=0.0)
    y = (X @ u.T - (c * u).sum(1)[None, :]) / sigma
    G = torch.exp(-r2 / (2 * sigma * sigma))
    out = torch.zeros_like(G)
    for a in range(int(o.max()) + 1):
        mask = torch.as_tensor(o == a)
        if mask.any():
            out[:, mask] = ((-1) ** a) * sigma ** (-a) * _he(a, y[:, mask]) * G[:, mask]
    return out


def mp_inner(A, B, sigma, d, dps=50):
    """sum_{p in A, q in B} w_p w_q (prim_p, prim_q)_{H^1} in mpmath; A, B are lists of (c, u, a, w)."""
    with mp.workdps(dps):
        sg = mp.mpf(sigma)
        tau = mp.sqrt(2) * sg
        C = (mp.pi * sg * sg) ** (mp.mpf(d) / 2)
        total = mp.mpf(0)
        for (ca, ua, a, wa) in A:
            ca = [mp.mpf(float(x)) for x in ca]; ua = [mp.mpf(float(x)) for x in ua]
            for (cb, ub, b, wb) in B:
                cbm = [mp.mpf(float(x)) for x in cb]; ubm = [mp.mpf(float(x)) for x in ub]
                z = [x - y for x, y in zip(ca, cbm)]
                r2 = mp.fsum(x * x for x in z)
                alpha = mp.fsum(x * y for x, y in zip(ua, z))
                beta = mp.fsum(x * y for x, y in zip(ubm, z))
                rho = mp.fsum(x * y for x, y in zip(ua, ubm))
                c = _k_coeffs(alpha, beta, rho, r2, tau, C, a, b)
                P0 = 1 + mp.mpf(d) / tau ** 2 - r2 / tau ** 4
                K = _K_coeff(c, a, b, alpha, beta, rho, P0, tau)
                total += mp.mpf(float(wa)) * mp.mpf(float(wb)) * ((-1) ** a) * FACT[a] * FACT[b] * K
        return total


def mp_norm2(A, sigma, d, dps=50):
    """||sum_{p in A} w_p prim_p||_{H^1}^2 in mpmath (symmetric version of mp_inner)."""
    with mp.workdps(dps):
        sg = mp.mpf(sigma)
        tau = mp.sqrt(2) * sg
        C = (mp.pi * sg * sg) ** (mp.mpf(d) / 2)
        P = [([mp.mpf(float(x)) for x in c], [mp.mpf(float(x)) for x in u], int(a), mp.mpf(float(w))) for c, u, a, w in A]
        total = mp.mpf(0)
        for i, (ca, ua, a, wa) in enumerate(P):
            for j in range(i, len(P)):
                cb, ub, b, wb = P[j]
                z = [x - y for x, y in zip(ca, cb)]
                r2 = mp.fsum(x * x for x in z)
                alpha = mp.fsum(x * y for x, y in zip(ua, z))
                beta = mp.fsum(x * y for x, y in zip(ub, z))
                rho = mp.fsum(x * y for x, y in zip(ua, ub))
                c = _k_coeffs(alpha, beta, rho, r2, tau, C, a, b)
                P0 = 1 + mp.mpf(d) / tau ** 2 - r2 / tau ** 4
                K = _K_coeff(c, a, b, alpha, beta, rho, P0, tau)
                term = wa * wb * ((-1) ** a) * FACT[a] * FACT[b] * K
                total += term if j == i else 2 * term
        return total


def derivative_norm(order, sigma, d):
    """||d_e^order G||_{H^1} for a unit vector e."""
    c = torch.zeros(1, d, dtype=torch.float64)
    u = torch.zeros(1, d, dtype=torch.float64); u[0, 0] = 1.0
    o = np.array([order])
    return float(torch.sqrt(h1_gram(c, u, o, c, u, o, sigma))[0, 0])
