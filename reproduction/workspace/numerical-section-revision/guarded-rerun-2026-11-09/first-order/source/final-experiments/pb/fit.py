"""Unconstrained Newton coefficient fit with cached-node backtracking.

This minimizes the original quadratic-plus-PB objective over all coefficients.
No coefficient constraint, ridge, field clipping, or Tikhonov term is added.
Rejected Armijo trials evaluate only nonlinear values.  Objective differences
use the quadratic increment formula rather than subtracting two large offset
objectives.  Target transcendental functions are cached within each fit.
"""
from __future__ import annotations

import math

import numpy as np

from quadrature import _small_remainders, _half_exp_bregman, bregman_parts


class _Nonlinear:
    def __init__(self, Phi, U, W):
        self.Phi, self.U, self.W = Phi, U, W
        self.sinh_u, self.cosh_u = np.sinh(U), np.cosh(U)
        self.cosh_u_minus_one = 2 * np.sinh(U / 2) ** 2
        self.value_calls = self.derivative_calls = 0

    def gap(self, V):
        self.value_calls += 1
        delta = V - self.U
        if not np.all(np.isfinite(delta)):
            return np.full_like(delta, np.inf)
        if np.max(np.abs(delta)) <= 0.5:
            sr, cr = _small_remainders(delta)
            gap = (0.5 * delta * delta * self.cosh_u_minus_one
                   + self.cosh_u * cr + self.sinh_u * sr)
        else:
            # General trials still compute values only.  Factored exponential
            # remainders avoid cancellation for opposite-sign large fields.
            near = np.abs(delta) <= 0.5
            gap = np.empty_like(delta)
            d = delta[near]
            sr, cr = _small_remainders(d)
            gap[near] = (0.5 * d * d * self.cosh_u_minus_one[near]
                         + self.cosh_u[near] * cr + self.sinh_u[near] * sr)
            d, a = delta[~near], self.U[~near]
            with np.errstate(over="ignore", invalid="ignore"):
                gap[~near] = (_half_exp_bregman(a, d) + _half_exp_bregman(-a, -d)
                              - 0.5 * d * d)
            gap[~np.isfinite(gap)] = np.inf
        return gap

    def derivatives(self, V):
        self.derivative_calls += 1
        delta = V - self.U
        if np.all(np.isfinite(delta)) and np.max(np.abs(delta)) <= 0.5:
            sr, _ = _small_remainders(delta)
            residual = (self.sinh_u * (2 * np.sinh(delta / 2) ** 2)
                        + self.cosh_u * sr + self.cosh_u_minus_one * delta)
            curvature = 2 * np.sinh(V / 2) ** 2
        else:
            _, residual, curvature = bregman_parts(V, self.U)
        return (self.Phi.T @ (self.W * residual),
                self.Phi.T @ ((self.W * curvature)[:, None] * self.Phi))


def solve_unconstrained_quadratic(M, b):
    """Solve the full scaled linear system; never regularize or truncate rank."""
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.all(np.isfinite(M)) or np.any(np.diag(M) <= 0):
        raise FloatingPointError("nonpositive or nonfinite Newton diagonal")
    scale = np.sqrt(np.diag(M))
    scaled = M / scale[:, None] / scale[None, :]
    rhs = b / scale
    try:
        coefficient = np.linalg.solve(scaled, rhs)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("singular unconstrained coefficient system") from exc
    residual = scaled @ coefficient - rhs
    denominator = np.linalg.norm(scaled, ord=np.inf) * np.linalg.norm(coefficient, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
    if not np.all(np.isfinite(coefficient)) or np.linalg.norm(residual, ord=np.inf) > 1e-10 * max(denominator, np.finfo(float).tiny):
        raise FloatingPointError("unreliable unconstrained coefficient solve")
    return coefficient / scale


def _kkt(M, b, rho, C, g, nonlinear_gradient, scale):
    gradient = M @ g - b + nonlinear_gradient
    return {"kkt_residual": float(np.max(np.abs(gradient))),
            "kkt_scaled_residual": float(np.max(np.abs(gradient) / scale)),
            "multiplier": 0.0, "cost": float(rho @ np.abs(g)),
            "coefficient_constraint": None, "budget_violation": None,
            "complementarity": None}


def solve_coefficients(M, b, rho, C, Phi, U, W, g0, max_newton=60, tol=1e-13):
    """Return coefficients and diagnostics for the unchanged PB objective.

    The quadratic Newton model uses a full column-scaled linear solve.
    The Armijo coefficient (1e-4), fallback acceptance of a nonincreasing step,
    smallest tested step (above 1e-8), Newton cap, and full-step relative-change
    tolerance retain the original Newton protocol.  Relative changes are evaluated from
    stable increments, with the same omitted-target-constant objective scale.

    An additional sufficient stopping test uses
      ||gradient / sqrt(diag(M))||² / (2 lambda_min(D^-1 M D^-1)).
    This bounds the gap to the unrestricted minimum, because the nonlinear Hessian is positive semidefinite.  The test
    is used only when the normalized Gram eigenvalue is comfortably positive,
    and its threshold is tol**2 times the objective scale.  Squaring tol avoids
    terminating a warm start at an O(sqrt(tol)) coefficient-error floor before
    a full Newton correction is taken.
    It is an exact-arithmetic convexity bound, not a validated rounding bound.
    No small-step or decrement test is substituted for the energy gate.

    A finite coefficient warm start can overflow the nonlinear energy when
    transferred to a new geometry.  In that case only the initializer is
    discarded: restart from the feasible zero vector and solve the same
    unconstrained objective with the same stopping rules.
    """
    M, b, rho, Phi, U, W = (np.asarray(x, dtype=float) for x in (M, b, rho, Phi, U, W))
    g = np.asarray(g0, dtype=float).copy()
    if not np.all(np.isfinite(g)):
        raise ValueError("initial coefficient vector must be finite")
    if C is not None:
        raise ValueError("this solver requires an unconstrained coefficient problem (C=None)")
    scale = np.sqrt(np.maximum(np.diag(M), 1e-300))
    eigenvalues = np.linalg.eigvalsh(M / scale[:, None] / scale[None, :])
    lower = float(eigenvalues[0])
    reliable_lower = lower > 128 * np.finfo(float).eps * max(1.0, float(eigenvalues[-1]))
    nonlin = _Nonlinear(Phi, U, W)
    with np.errstate(over="ignore", invalid="ignore"):
        V = Phi @ g
        gap = nonlin.gap(V)
        nonlinear_value = float(W @ gap)
    initial_guess_reset = not (np.all(np.isfinite(V))
                               and np.all(np.isfinite(gap))
                               and math.isfinite(nonlinear_value))
    if initial_guess_reset:
        g = np.zeros_like(g)
        V = np.zeros_like(U)
        gap = nonlin.gap(V)
        nonlinear_value = float(W @ gap)
        if not np.all(np.isfinite(gap)) or not math.isfinite(nonlinear_value):
            raise FloatingPointError("nonfinite zero-initialized nonlinear coefficient energy")
    grad_n, hess_n = nonlin.derivatives(V)
    f = float(0.5 * g @ M @ g - b @ g + nonlinear_value)
    qp_solves = backtracks = 0
    reason = "newton_limit"
    gap_bound, decrement = math.inf, math.inf
    k = -1
    for k in range(max_newton):
        grad_quad = M @ g - b
        gradient = grad_quad + grad_n
        objective_scale = max(1.0, abs(f))
        gap_bound = (float((gradient / scale) @ (gradient / scale)) / (2 * lower)
                     if reliable_lower else math.inf)
        if gap_bound <= tol * tol * objective_scale:
            reason = "quadratic_gradient_gap_bound"
            break
        H = M + hess_n
        H = 0.5 * (H + H.T)
        lin = b + hess_n @ g - grad_n
        g_new = solve_unconstrained_quadratic(H, lin)
        qp_solves += 1
        p = g_new - g
        if np.max(np.abs(p)) <= 1e-15 * max(1.0, float(np.max(np.abs(g)))):
            reason = "coefficient_step"
            break
        decrease = float(gradient @ p)
        pMp = float(p @ M @ p)
        linear_quad_change = float(grad_quad @ p)
        decrement = max(0.0, -decrease)
        delta_V = Phi @ p
        alpha = 1.0
        accepted = False
        while alpha > 1e-8:
            trial_gap = nonlin.gap(V + alpha * delta_V)
            if np.all(np.isfinite(trial_gap)):
                nonlinear_change = float(W @ (trial_gap - gap))
                change = alpha * linear_quad_change + 0.5 * alpha * alpha * pMp + nonlinear_change
            else:
                change = math.inf
            if change <= 1e-4 * alpha * decrease or change <= 0.0:
                accepted = True
                break
            backtracks += 1
            alpha *= 0.5
        if not accepted:
            reason = "line_search_no_decrease"
            break
        g = g + alpha * p
        # Refresh the represented field after acceptance to avoid accumulating
        # summation drift in ill-conditioned ordinary coordinates.
        V = Phi @ g
        gap = nonlin.gap(V)
        grad_n, hess_n = nonlin.derivatives(V)
        f += change
        if abs(change) / max(1.0, abs(f)) <= tol and alpha == 1.0:
            reason = "full_step_energy_change"
            break
    gradient = M @ g - b + grad_n
    final_bound = (float((gradient / scale) @ (gradient / scale)) / (2 * lower)
                   if reliable_lower else math.inf)
    info = {"newton": k + 1, "qp_solves": qp_solves,
            "initial_guess_reset": bool(initial_guess_reset),
            "nonlinearvals": nonlin.value_calls, "nonlinear_derivatives": nonlin.derivative_calls,
            "backtracks": backtracks, "stop": reason,
            "newton_decrement_squared": decrement,
            "quadratic_gradient_gap_bound": final_bound,
            "gram_normalized_min_eigenvalue": lower,
            **_kkt(M, b, rho, C, g, grad_n, scale)}
    return g, info
