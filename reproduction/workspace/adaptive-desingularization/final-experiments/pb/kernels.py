"""Dimension-independent Gaussian pair calculus for the PB benchmark.

All differentiable calculations use torch float64.  A primitive is
``a G(x-c) + v.dot(grad_x G(x-c))``.  Pair columns are E_h, D_h, with
``D_h = (G_{c+h e} - G_{c-h e})/(2h) -> -partial_e G_c``.

The H1 kernel is analytic.  Finite-pair Gram entries use a short, separately
validated one-dimensional Gauss--Legendre integral of derivative primitives;
pointwise pair evaluation and high-precision error evaluation are exact.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp
import numpy as np
import torch


def _tensor(value, like=None):
    return torch.as_tensor(value, dtype=torch.float64,
                           device=None if like is None else like.device)


def gaussian_gram(ca, va, aa, cb=None, vb=None, ab=None, sigma=0.45):
    """H1(R^d) Gram matrix for Gaussian/first-derivative primitives.

    Centers and spatial derivative vectors have shapes (n,d); amplitudes
    have shape (n,).  Omitting the second family gives a self Gram matrix.
    Gradients propagate through centers, vectors, and amplitudes.
    """
    ca = _tensor(ca)
    va, aa = _tensor(va, ca), _tensor(aa, ca)
    if cb is None:
        cb, vb, ab = ca, va, aa
    else:
        cb, vb, ab = _tensor(cb, ca), _tensor(vb, ca), _tensor(ab, ca)
    z = ca[:, None, :] - cb[None, :, :]
    r = (z * z).sum(dim=-1)
    d = ca.shape[-1]
    s2 = sigma * sigma
    alpha, beta = 1.0 / (4.0 * s2), 1.0 / (4.0 * s2 * s2)
    const = math.pi ** (d / 2.0) * sigma ** d
    pref = const * torch.exp(-alpha * r)
    poly = 1.0 + d / (2.0 * s2) - beta * r
    f0 = pref * poly
    f1 = pref * (-alpha * poly - beta)
    f2 = pref * (alpha * alpha * poly + 2.0 * alpha * beta)
    za = (z * va[:, None, :]).sum(dim=-1)
    zb = (z * vb[None, :, :]).sum(dim=-1)
    vv = va @ vb.T
    return (aa[:, None] * ab[None, :] * f0
            + 2.0 * f1 * (aa[:, None] * zb - ab[None, :] * za)
            - 4.0 * f2 * za * zb - 2.0 * f1 * vv)


@lru_cache(maxsize=None)
def _legendre(order):
    if order < 1:
        raise ValueError("pair quadrature order must be positive")
    t, w = np.polynomial.legendre.leggauss(order)
    return t, 0.5 * w


def basis_primitives(blocks, sigma=0.45, order=8):
    """Return ``(centers, vectors, amplitudes, T)`` for all block columns.

    A block is a dict containing ``kind`` ('single'/'pair'), ``c`` (d,),
    and, for a pair, unit ``e`` (d,), scalar ``h``, and ``completed`` bool.
    Column ordering is one G column per singleton and E,D per pair.
    ``T`` maps column coefficients to primitive coefficients.  For finite
    pairs D is represented by -1/2 integral partial_e G_{c+t h e} dt.
    Completed pairs use G and -partial_e G directly, with no quadrature.
    """
    del sigma  # Geometry of the representation does not depend on the width.
    if not blocks:
        raise ValueError("at least one block is required")
    reference = _tensor(blocks[0]["c"])
    nodes_np, weights_np = _legendre(int(order))
    nodes, weights = _tensor(nodes_np, reference), _tensor(weights_np, reference)
    centers, vectors, amplitudes, entries = [], [], [], []
    ncols = 0

    def add(c, v, a, column, weight=1.0):
        row = len(centers)
        centers.append(c)
        vectors.append(v)
        amplitudes.append(a)
        entries.append((row, column, weight))

    for block in blocks:
        c = _tensor(block["c"], reference)
        zero = torch.zeros_like(c)
        if block["kind"] == "single":
            add(c, zero, 1.0, ncols)
            ncols += 1
            continue
        if block["kind"] != "pair":
            raise ValueError("block kind must be 'single' or 'pair'")
        e = _tensor(block["e"], reference)
        if block.get("completed", False):
            add(c, zero, 1.0, ncols)
            add(c, -e, 0.0, ncols + 1)
        else:
            h = _tensor(block["h"], reference)
            add(c + h * e, zero, 1.0, ncols, 0.5)
            add(c - h * e, zero, 1.0, ncols, 0.5)
            for t, w in zip(nodes, weights):
                add(c + t * h * e, -e, 0.0, ncols + 1, w)
        ncols += 2
    T = torch.zeros((len(centers), ncols), dtype=torch.float64, device=reference.device)
    for row, col, weight in entries:
        T[row, col] = weight
    return (torch.stack(centers), torch.stack(vectors),
            _tensor(amplitudes, reference), T)


def block_gram(blocks, other_blocks=None, sigma=0.45, order=8):
    """Gram matrix of the stable block columns, retaining autograd."""
    ca, va, aa, Ta = basis_primitives(blocks, sigma, order)
    if other_blocks is None:
        return Ta.T @ gaussian_gram(ca, va, aa, sigma=sigma) @ Ta
    cb, vb, ab, Tb = basis_primitives(other_blocks, sigma, order)
    return Ta.T @ gaussian_gram(ca, va, aa, cb, vb, ab, sigma) @ Tb


def coefficient_primitives(blocks, coeffs, sigma=0.45, order=8):
    """Combine block coefficients into primitive amplitudes and vectors.

    Useful for a completed manufactured target: the returned three arrays
    can be passed directly as either family to ``gaussian_gram``.
    """
    c, v, a, T = basis_primitives(blocks, sigma, order)
    weights = T @ _tensor(coeffs, c)
    return c, v * weights[:, None], a * weights


def target_primitives(blocks, coeffs, sigma=0.45):
    """Primitive representation of a target made of completed pairs/singles."""
    if any(b["kind"] == "pair" and not b.get("completed", False) for b in blocks):
        raise ValueError("target_primitives expects completed target pairs")
    return coefficient_primitives(blocks, coeffs, sigma)


def _phi(t):
    """(1-exp(-t))/t, including its smooth value and derivative at zero."""
    small = torch.abs(t) < 1e-4
    safe_t = torch.where(small, torch.ones_like(t), t)
    exact = -torch.expm1(-t) / safe_t
    series = 1.0 + t * (-0.5 + t * (1.0 / 6.0 + t * (-1.0 / 24.0
                   + t * (1.0 / 120.0 + t * (-1.0 / 720.0)))))
    return torch.where(small, series, exact)


def column_values(blocks, X, sigma=0.45):
    """Exact, cancellation-free values of all columns at points X (n,d).

    The finite-pair formulas use a bounded exponential and expm1.  They
    remain smooth for automatic differentiation at h=0 and e.(x-c)=0.
    Direction vectors must be unit length, as in ``basis_primitives``.
    """
    X = _tensor(X)
    s2 = sigma * sigma
    cols = []
    for block in blocks:
        c = _tensor(block["c"], X)
        y = X - c
        r = (y * y).sum(dim=-1)
        if block["kind"] == "single":
            cols.append(torch.exp(-r / (2.0 * s2)))
            continue
        e = _tensor(block["e"], X)
        s = y @ e
        if block.get("completed", False):
            g = torch.exp(-r / (2.0 * s2))
            cols.extend((g, g * s / s2))
            continue
        h = _tensor(block["h"], X)
        q = torch.abs(h * s / s2)
        peak = torch.exp(-(r + h * h) / (2.0 * s2) + q)
        cols.extend((0.5 * peak * (1.0 + torch.exp(-2.0 * q)),
                     peak * (s / s2) * _phi(2.0 * q)))
    return torch.stack(cols, dim=-1)


def kappa(d, k, sigma=0.45):
    """Return ||partial_e^k G_sigma||_{H1(R^d)} for any unit direction e."""
    if d < 1 or k < 0 or int(k) != k:
        raise ValueError("d must be positive and k a nonnegative integer")
    k = int(k)
    log_sq = (0.5 * d * math.log(math.pi) + d * math.log(sigma)
              + math.lgamma(2 * k + 1) - k * math.log(4.0)
              - math.lgamma(k + 1) - 2 * k * math.log(sigma)
              + math.log1p((d + 2 * k) / (2.0 * sigma * sigma)))
    return math.exp(0.5 * log_sq)


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _mp_primitives(blocks, coeffs):
    """Exact ordinary/confluent primitives using arbitrary precision."""
    coeffs = _plain(coeffs).reshape(-1)
    result = []
    offset = 0
    for block in blocks:
        c = [mp.mpf(float(x)) for x in _plain(block["c"]).reshape(-1)]
        a = mp.mpf(float(coeffs[offset]))
        zero = [mp.mpf(0) for _ in c]
        if block["kind"] == "single":
            result.append((c, zero, a))
            offset += 1
            continue
        b = mp.mpf(float(coeffs[offset + 1]))
        e = [mp.mpf(float(x)) for x in _plain(block["e"]).reshape(-1)]
        h = mp.mpf(float(_plain(block.get("h", 0.0))))
        if block.get("completed", False) or h == 0:
            result.append((c, [-b * x for x in e], a))
        else:
            result.append(([x + h * v for x, v in zip(c, e)], zero, a / 2 + b / (2 * h)))
            result.append(([x - h * v for x, v in zip(c, e)], zero, a / 2 - b / (2 * h)))
        offset += 2
    if offset != len(coeffs):
        raise ValueError("coefficient count does not match the blocks")
    return result


def _mp_inner(A, B, sigma):
    d = len(A[0][0])
    s = mp.mpf(float(sigma))
    alpha, beta = 1 / (4 * s * s), 1 / (4 * s ** 4)
    const = mp.pi ** (mp.mpf(d) / 2) * s ** d
    total = mp.mpf(0)
    for ca, va, aa in A:
        for cb, vb, ab in B:
            z = [x - y for x, y in zip(ca, cb)]
            r = sum(x * x for x in z)
            pref = const * mp.exp(-alpha * r)
            poly = 1 + d / (2 * s * s) - beta * r
            f0 = pref * poly
            f1 = pref * (-alpha * poly - beta)
            f2 = pref * (alpha * alpha * poly + 2 * alpha * beta)
            za = sum(x * y for x, y in zip(z, va))
            zb = sum(x * y for x, y in zip(z, vb))
            vv = sum(x * y for x, y in zip(va, vb))
            total += aa * ab * f0 + 2 * f1 * (aa * zb - ab * za) - 4 * f2 * za * zb - 2 * f1 * vv
    return total


def high_precision_error(blocks, coeffs, target_blocks, target_coeffs,
                         sigma=0.45, dps=80):
    """Exact H1 error, relative error, and target norm using mpmath.

    Finite pairs are evaluated as two ordinary atoms at arbitrary precision,
    so this metric does not inherit the short pair quadrature error used in
    optimization.  Input float64 parameters are promoted exactly to mpmath.
    Returns a dict with float values ``norm``, ``relative``, ``target_norm``.
    """
    with mp.workdps(dps):
        A = _mp_primitives(blocks, coeffs)
        B = _mp_primitives(target_blocks, target_coeffs)
        difference = A + [(c, [-x for x in v], -a) for c, v, a in B]
        norm2 = _mp_inner(difference, difference, sigma)
        target2 = _mp_inner(B, B, sigma)
        if norm2 < 0:
            if abs(norm2) > mp.power(10, -dps // 2) * max(1, target2):
                raise ArithmeticError("negative high-precision squared H1 error")
            norm2 = mp.mpf(0)
        norm, target_norm = mp.sqrt(norm2), mp.sqrt(target2)
        return {"norm": float(norm), "relative": float(norm / target_norm),
                "target_norm": float(target_norm)}


def validation_checks():
    """Reproducible checks of pair values, Gram integration, and gradients."""
    rng = np.random.default_rng(23)
    value_error = gram_error = exact_error = 0.0
    for d in (2, 3, 4, 5):
        blocks = []
        for _ in range(4):
            e = rng.normal(size=d)
            e /= np.linalg.norm(e)
            blocks.append(dict(kind="pair", c=rng.normal(size=d) * 0.3,
                               e=e, h=0.27, completed=False))
        X = _tensor(rng.normal(size=(20, d)))
        actual = column_values(blocks, X)
        expected = []
        for b in blocks:
            c, e, h = _tensor(b["c"]), _tensor(b["e"]), b["h"]
            gp = torch.exp(-((X - c - h * e) ** 2).sum(-1) / (2 * 0.45 ** 2))
            gm = torch.exp(-((X - c + h * e) ** 2).sum(-1) / (2 * 0.45 ** 2))
            expected.extend(((gp + gm) / 2, (gp - gm) / (2 * h)))
        value_error = max(value_error, float((actual - torch.stack(expected, -1)).abs().max()))
        M, M16 = block_gram(blocks), block_gram(blocks, order=16)
        gram_error = max(gram_error, float((M - M16).abs().max()))
        co = rng.normal(size=8)
        with mp.workdps(80):
            primitives = _mp_primitives(blocks, co)
            exact = float(_mp_inner(primitives, primitives, 0.45))
        exact_error = max(exact_error, abs(exact - float(_tensor(co) @ M @ _tensor(co))))
        b0 = dict(blocks[0], completed=True, h=0.0)
        G = block_gram([b0])
        assert abs(float(G[0, 0]) - kappa(d, 0) ** 2) < 1e-12
        assert abs(float(G[1, 1]) - kappa(d, 1) ** 2) < 1e-12
        assert high_precision_error([b0], [1, 0.3], [b0], [1, 0.3])["relative"] == 0
    d = 5
    p = _tensor(rng.normal(size=2 * d + 1))
    p[-1] = 0.08
    p.requires_grad_(True)
    X = _tensor(rng.normal(size=(20, d)))

    def objective(z):
        e = z[d:2 * d] / z[d:2 * d].norm()
        blocks = [dict(kind="pair", c=z[:d], e=e, h=z[-1], completed=False)]
        return (column_values(blocks, X) ** 2).sum() + block_gram(blocks).sum()

    grad = torch.autograd.grad(objective(p), p)[0]
    directions = torch.eye(len(p), dtype=torch.float64)
    step = 1e-5
    fd = torch.stack([(objective(p.detach() + step * v) - objective(p.detach() - step * v))
                      / (2 * step) for v in directions])
    grad_error = float((fd - grad).norm() / grad.norm())
    assert value_error < 1e-13
    assert gram_error < 1e-12
    assert exact_error < 2e-11
    assert grad_error < 1e-7
    return dict(max_pair_value_error=value_error, max_pair_gram_order_error=gram_error,
                max_exact_quadratic_error=exact_error, gradient_relative_error=grad_error)


if __name__ == "__main__":
    import json
    torch.set_num_threads(1)
    print(json.dumps(validation_checks(), indent=2))
