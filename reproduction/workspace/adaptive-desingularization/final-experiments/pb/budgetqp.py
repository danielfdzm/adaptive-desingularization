"""Convex coefficient fitting under a weighted l1 budget.

    minimize  1/2 g^T M g - b^T g  (+ optional convex nonlinearity)   s.t.  sum_i rho_i |g_i| <= C.

Quadratic case: FISTA on the rescaled l1 ball to identify the active set and signs, followed by an
exact KKT polish on the active set and a KKT verification.  Dimension is tiny (<= a few dozen),
so every operation is dense.
"""
from __future__ import annotations

import numpy as np


def project_l1_ball(y: np.ndarray, C: float) -> np.ndarray:
    """Euclidean projection of y onto {x : ||x||_1 <= C}."""
    a = np.abs(y)
    if a.sum() <= C:
        return y.copy()
    u = np.sort(a)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(u) + 1)
    cond = u - (css - C) / k > 0
    rho_idx = np.nonzero(cond)[0][-1]
    theta = (css[rho_idx] - C) / (rho_idx + 1)
    return np.sign(y) * np.maximum(a - theta, 0.0)


def kkt_polish(M, b, rho, C, active, signs):
    """Solve the equality constrained QP on the active set:  M_AA g_A + lam * rho_A s_A = b_A,
    sum_A rho_A s_A g_A = C.  Returns (g, lam) or None if singular."""
    n = len(b)
    A = np.nonzero(active)[0]
    if len(A) == 0:
        return np.zeros(n), 0.0
    MAA = M[np.ix_(A, A)]
    w = rho[A] * signs[A]
    K = np.zeros((len(A) + 1, len(A) + 1))
    K[: len(A), : len(A)] = MAA
    K[: len(A), -1] = w
    K[-1, : len(A)] = w
    rhs = np.concatenate([b[A], [C]])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        return None
    g = np.zeros(n)
    g[A] = sol[: len(A)]
    return g, float(sol[-1])


def solve_budget_qp(M: np.ndarray, b: np.ndarray, rho: np.ndarray, C: float, g0=None,
                    max_iter: int = 20000, tol: float = 1e-13):
    """Return (g, info) minimizing 1/2 g^T M g - b^T g subject to sum rho_i |g_i| <= C."""
    n = len(b)
    rho = np.asarray(rho, dtype=float)
    # unconstrained minimizer
    d = np.sqrt(np.maximum(np.diag(M), 1e-300))
    Mn = M / d[:, None] / d[None, :]
    try:
        gu = np.linalg.solve(Mn, b / d) / d
        unconstrained_ok = np.all(np.isfinite(gu))
    except np.linalg.LinAlgError:
        gu = None
        unconstrained_ok = False
    if unconstrained_ok and np.sum(rho * np.abs(gu)) <= C * (1 + 1e-14):
        return gu, {"active": int(np.count_nonzero(gu)), "lam": 0.0, "constrained": False, "iters": 0}

    # rescale: y = rho * g ;  objective 1/2 y^T Mt y - bt^T y,  ||y||_1 <= C
    Dinv = 1.0 / rho
    Mt = (M * Dinv[:, None]) * Dinv[None, :]
    bt = b * Dinv
    L = np.linalg.eigvalsh(Mt).max()
    mu = max(np.linalg.eigvalsh(Mt).min(), 0.0)
    step = 1.0 / L
    if g0 is not None and np.all(np.isfinite(g0)):
        y = project_l1_ball(np.asarray(g0, float) * rho, C)
    else:
        y = project_l1_ball(bt / L, C)
    z = y.copy()
    t_k = 1.0
    best = None
    it = 0
    obj = lambda yy: 0.5 * yy @ Mt @ yy - bt @ yy
    f_old = obj(y)
    polish_every = 25
    for it in range(1, max_iter + 1):
        grad = Mt @ z - bt
        y_new = project_l1_ball(z - step * grad, C)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        z = y_new + ((t_k - 1.0) / t_new) * (y_new - y)
        # adaptive restart (O'Donoghue-Candes)
        if (y_new - y) @ grad > 0:
            z = y_new.copy()
            t_new = 1.0
        y, t_k = y_new, t_new
        if it % polish_every == 0:
            active = np.abs(y) > 0
            signs = np.sign(y)
            res = kkt_polish(M, b, rho, C, active, signs)
            if res is not None:
                g, lam = res
                if lam >= -1e-12 and np.all(g[active] * signs[active] >= -1e-15):
                    grad_g = M @ g - b
                    viol = np.abs(grad_g[~active]) - lam * rho[~active]
                    if (viol.size == 0 or viol.max() <= 1e-9 * max(1.0, lam) + 1e-13) and np.all(
                            np.abs(grad_g[active] + lam * rho[active] * signs[active]) <= 1e-9 * max(1.0, np.abs(grad_g).max())):
                        return g, {"active": int(active.sum()), "lam": lam, "constrained": True, "iters": it}
            f_new = obj(y)
            if abs(f_old - f_new) <= tol * max(1.0, abs(f_new)):
                break
            f_old = f_new
    # fall back to the iterate (project exactly onto the boundary)
    g = y * Dinv
    return g, {"active": int(np.count_nonzero(g)), "lam": float("nan"), "constrained": True, "iters": it, "fallback": True}


def solve_budget_convex(M, b, rho, C, nonlin, g0, max_newton: int = 60, tol: float = 1e-13):
    """Proximal Newton for  1/2 g^T M g - b^T g + N(g)  s.t. sum rho|g| <= C.

    nonlin(g) must return (value, gradient, hessian) of the convex term N.
    """
    g = np.asarray(g0, dtype=float).copy()
    if np.sum(rho * np.abs(g)) > C:
        g = project_l1_ball(g * rho, C) / rho
    val, grad_n, hess_n = nonlin(g)
    f = 0.5 * g @ M @ g - b @ g + val
    nfit = 0
    for k in range(max_newton):
        H = M + hess_n
        H = 0.5 * (H + H.T)
        # quadratic model around g:  1/2 (g+p)^T H (g+p) - (H g - (M g - b + grad_n))^T (g+p)
        lin = H @ g - (M @ g - b + grad_n)
        g_new, info = solve_budget_qp(H, lin, rho, C, g0=g)
        nfit += 1
        p = g_new - g
        if np.abs(p).max() <= 1e-15 * max(1.0, np.abs(g).max()):
            break
        # backtracking on the true objective
        alpha = 1.0
        decrease = (M @ g - b + grad_n) @ p
        while alpha > 1e-8:
            gt = g + alpha * p
            valt, gradt, hesst = nonlin(gt)
            ft = 0.5 * gt @ M @ gt - b @ gt + valt
            if ft <= f + 1e-4 * alpha * decrease or ft <= f:
                break
            alpha *= 0.5
        if ft > f:
            break
        rel = abs(f - ft) / max(1.0, abs(f))
        g, f, grad_n, hess_n = gt, ft, gradt, hesst
        if rel <= tol and alpha == 1.0:
            break
    return g, {"newton": k + 1, "qp_solves": nfit}
