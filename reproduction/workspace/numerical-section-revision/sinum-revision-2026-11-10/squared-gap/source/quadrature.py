"""Positive whole-space quadrature for the higher-dimensional PB benchmark.

Each of the K target centers contributes 2**power points from an independent
scrambled Sobol rule, transformed to N(center, sigma**2 I / 2).  If q is the
equally weighted mixture of these K Gaussian densities, the integration weight
at every point is 1 / (K * 2**power * q(point)).  These weights are positive;
in particular, quadrature preserves convexity of the coefficient objective.

Use the same training rule for both methods.  Check final states and completion
energy changes on independent, more highly resolved rules.  ``validate_gap``
provides that comparison, including replicate spread; it does not claim a
rigorous integration error bound.  No finite-box truncation is introduced.

Example::

    X, weights, metadata = make_quadrature(centers, power=12, seed=20261027)
    U = evaluate_target(X)
    value = weighted_bregman(evaluate_model(X), U, weights)
    check = validate_gap(centers, evaluate_model, evaluate_target,
                         powers=(12, 14, 16), seeds=(9101, 9102, 9103))

The analytic quadratic contribution 0.5 * ||v-u*||_H1**2 must be added by the
caller.  All functions below concern N(z)=cosh(z)-1-z**2/2 only.  The manufactured
target is stationary for every such rule because the nonlinear load and trial
state are evaluated at the same points.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


def make_quadrature(
    centers: np.ndarray,
    sigma: float = 0.45,
    power: int = 12,
    seed: int = 20261027,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return nodes, Lebesgue-integration weights, and reproducibility metadata.

    ``centers`` has shape (K, d), with K=6 in the paper benchmark.  The returned
    arrays have shapes (K * 2**power, d) and (K * 2**power,).  The weights do not
    sum to one: they integrate functions against Lebesgue measure on R**d.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or min(centers.shape) < 1 or not np.all(np.isfinite(centers)):
        raise ValueError("centers must be a nonempty finite array of shape (K, d)")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not isinstance(power, (int, np.integer)) or not 0 <= power <= 25:
        raise ValueError("power must be an integer between 0 and 25")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    count, dimension = centers.shape
    nodes_per_component = 2 ** int(power)
    total_nodes = count * nodes_per_component
    proposal_std = sigma / math.sqrt(2.0)
    child_seeds = [int(s.generate_state(1)[0])
                   for s in np.random.SeedSequence(int(seed)).spawn(count)]
    points = np.empty((total_nodes, dimension))
    endpoint_adjustments = 0
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    for k, (center, child_seed) in enumerate(zip(centers, child_seeds)):
        unit = qmc.Sobol(dimension, scramble=True, seed=child_seed).random_base2(int(power))
        endpoint_adjustments += int(np.count_nonzero((unit == 0.0) | (unit == 1.0)))
        unit = np.clip(unit, lower, upper)
        points[k * nodes_per_component:(k + 1) * nodes_per_component] = (
            center + proposal_std * ndtri(unit)
        )

    # Chunking avoids an N x K x d temporary when independent validation uses
    # millions of points.  The normalizing constant is essential: q is a density.
    weights = np.empty(total_nodes)
    density_log_constant = dimension * math.log(proposal_std * math.sqrt(2.0 * math.pi))
    for first in range(0, total_nodes, 65536):
        stop = min(first + 65536, total_nodes)
        offsets = points[first:stop, None, :] - centers[None, :, :]
        squared_distances = np.einsum("nkd,nkd->nk", offsets, offsets)
        log_density = (logsumexp(-squared_distances / (2.0 * proposal_std ** 2), axis=1)
                       - math.log(count) - density_log_constant)
        weights[first:stop] = np.exp(-math.log(total_nodes) - log_density)
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise FloatingPointError("importance weights must be finite and strictly positive")

    metadata = {
        "rule": "stratified_scrambled_sobol_gaussian_mixture",
        "domain": "R^d",
        "dimension": int(dimension),
        "components": int(count),
        "component_centers": centers.tolist(),
        "component_covariance_scalar": float(proposal_std ** 2),
        "component_weights": [1.0 / count] * count,
        "sigma": float(sigma),
        "power": int(power),
        "nodes_per_component": int(nodes_per_component),
        "nodes": int(total_nodes),
        "seed": int(seed),
        "component_seeds": child_seeds,
        "sobol_scramble": True,
        "inverse_cdf_endpoint_adjustments": endpoint_adjustments,
        "integration_weights": "1 / (nodes * mixture_density)",
        "positive_weights": True,
    }
    return points, weights, metadata


def _small_remainders(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sinh(x)-x and cosh(x)-1-x²/2, accurately for |x| <= 0.5."""
    x2 = x * x
    sinh_remainder = x * x2 * (
        1 / math.factorial(3) + x2 * (
            1 / math.factorial(5) + x2 * (
                1 / math.factorial(7) + x2 * (
                    1 / math.factorial(9) + x2 * (
                        1 / math.factorial(11) + x2 * (
                            1 / math.factorial(13) + x2 / math.factorial(15)))))))
    cosh_remainder = x2 * x2 * (
        1 / math.factorial(4) + x2 * (
            1 / math.factorial(6) + x2 * (
                1 / math.factorial(8) + x2 * (
                    1 / math.factorial(10) + x2 * (
                        1 / math.factorial(12) + x2 * (
                            1 / math.factorial(14) + x2 / math.factorial(16)))))))
    return sinh_remainder, cosh_remainder


def _half_exp_bregman(base: np.ndarray, difference: np.ndarray) -> np.ndarray:
    """0.5 exp(base) [exp(difference)-1-difference], without large products."""
    out = np.empty_like(base)
    forward = difference >= 0.0
    d = difference[forward]
    out[forward] = (np.exp(base[forward] + d - math.log(2.0))
                    * (-np.expm1(-d) - d * np.exp(-d)))
    d = difference[~forward]
    out[~forward] = np.exp(base[~forward] - math.log(2.0)) * (np.expm1(d) - d)
    return out


def bregman_parts(v, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nonlinear Bregman gap, v derivative, and v second derivative.

    Returned entries are, respectively,

        N(v)-N(u)-N'(u)*(v-u), N'(v)-N'(u), N''(v),

    for N(z)=cosh(z)-1-z²/2.  Inputs broadcast.  Small differences are evaluated
    with series; general differences use positive exponential remainders to
    avoid cancellation between opposite-sign states.  No field is clipped.
    Overflow/nonfinite trials yield an infinite gap and curvature, so callers
    can reject the trial; the residual retains its sign when possible.
    """
    v, u = np.broadcast_arrays(np.asarray(v, dtype=float), np.asarray(u, dtype=float))
    shape = v.shape
    vf, uf = v.ravel(), u.ravel()
    gap = np.full(vf.shape, np.inf)
    derivative = np.full(vf.shape, np.nan)
    curvature = np.full(vf.shape, np.inf)
    finite = np.isfinite(vf) & np.isfinite(uf)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        delta = vf[finite] - uf[finite]
        vv, uu = vf[finite], uf[finite]
        g = np.empty_like(delta)
        r = np.empty_like(delta)
        near = np.abs(delta) <= 0.5
        d, a = delta[near], uu[near]
        sr, cr = _small_remainders(d)
        cm1 = 2.0 * np.sinh(a / 2.0) ** 2
        g[near] = 0.5 * d * d * cm1 + np.cosh(a) * cr + np.sinh(a) * sr
        r[near] = np.sinh(a) * (2.0 * np.sinh(d / 2.0) ** 2) + np.cosh(a) * sr + cm1 * d
        d, a = delta[~near], uu[~near]
        g[~near] = (_half_exp_bregman(a, d) + _half_exp_bregman(-a, -d)
                    - 0.5 * d * d)
        r[~near] = np.sinh(vv[~near]) - np.sinh(a) - d
        h = 2.0 * np.sinh(vv / 2.0) ** 2
        # An arithmetic overflow in a trial is a failed trial, not a clipped
        # approximation of its objective.  NaNs from inf*0 follow the same rule.
        g[~np.isfinite(g)] = np.inf
        h[~np.isfinite(h)] = np.inf
        gap[finite], derivative[finite], curvature[finite] = g, r, h
    return gap.reshape(shape), derivative.reshape(shape), curvature.reshape(shape)


def weighted_bregman(v, u, weights) -> float:
    """Integrate only the nonlinear Bregman term; return inf for a failed trial."""
    gap, _, _ = bregman_parts(v, u)
    weights = np.asarray(weights, dtype=float)
    if gap.shape != weights.shape:
        raise ValueError("state values and weights must have the same shape")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("integration weights must be finite and positive")
    if not np.all(np.isfinite(gap)):
        return math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.dot(weights, gap))
    return result if math.isfinite(result) else math.inf


def weighted_coefficient_parts(
    coefficients: np.ndarray, features: np.ndarray, target_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Nonlinear value/gradient/Hessian for coefficient Newton fitting.

    ``features`` has shape (nodes, coefficients).  Add the analytic H¹
    quadratic value, gradient and Gram matrix in the caller.  Failed trials
    return an infinite value and NaN derivatives for explicit line-search
    rejection; this function never clips a trial or modifies its objective.
    """
    features = np.asarray(features, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    weights = np.asarray(weights, dtype=float)
    gap, residual, curvature = bregman_parts(features @ coefficients, target_values)
    if not np.all(np.isfinite(gap)) or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(curvature)):
        n = coefficients.size
        return math.inf, np.full(n, np.nan), np.full((n, n), np.nan)
    value = float(np.dot(weights, gap))
    gradient = features.T @ (weights * residual)
    hessian = features.T @ ((weights * curvature)[:, None] * features)
    return value, gradient, hessian


def validate_gap(
    centers: np.ndarray,
    evaluate_trial: Callable[[np.ndarray], np.ndarray],
    evaluate_target: Callable[[np.ndarray], np.ndarray],
    *,
    sigma: float = 0.45,
    powers: Sequence[int] = (12, 14, 16),
    seeds: Sequence[int] = (9101, 9102, 9103),
    evaluate_before: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Refine independent quadrature estimates of a gap or transfer change.

    With ``evaluate_before``, each estimate is nonlinear_gap(trial) minus
    nonlinear_gap(before), evaluated using common nodes.  Add the corresponding
    analytic quadratic change externally.  Replicate standard deviations and
    successive mean changes are empirical diagnostics, not error certificates.
    Seeds should differ from the training seed.  Larger powers with a fixed
    seed extend each component's scrambled Sobol sequence.
    """
    if not powers or len(seeds) < 2:
        raise ValueError("supply at least one power and two independent seeds")
    records = []
    previous_mean = None
    for power in powers:
        started = time.perf_counter()
        estimates = []
        for seed in seeds:
            points, weights, metadata = make_quadrature(centers, sigma, int(power), int(seed))
            target = evaluate_target(points)
            value = weighted_bregman(evaluate_trial(points), target, weights)
            if evaluate_before is not None:
                value -= weighted_bregman(evaluate_before(points), target, weights)
            estimates.append(value)
        mean = float(np.mean(estimates))
        records.append({
            "power": int(power), "nodes_per_replicate": metadata["nodes"],
            "seeds": [int(s) for s in seeds], "estimates": estimates,
            "mean": mean, "replicate_std": float(np.std(estimates, ddof=1)),
            "replicate_range": float(np.ptp(estimates)),
            "change_from_previous_mean": None if previous_mean is None else mean - previous_mean,
            "wall_seconds": time.perf_counter() - started,
        })
        previous_mean = mean
    return {"quantity": "nonlinear_transfer_change" if evaluate_before else "nonlinear_bregman_gap",
            "quadratic_part": "analytic; add separately", "rigorous_error_bound": False,
            "refinement": records}
