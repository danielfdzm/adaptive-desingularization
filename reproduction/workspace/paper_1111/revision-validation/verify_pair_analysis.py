#!/usr/bin/env python3
"""Independent high-precision checks for the pair-analysis revision.

These checks use the exact whole-space H1 Gaussian correlation, with no
quadrature or optimizer. They check constants and the artificial gap-bound
example. Algorithmic trajectories belong in a separate verification script.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 80


def correlation(z, dimension=1, sigma=mp.mpf(1)):
    return (
        mp.pi ** (mp.mpf(dimension) / 2)
        * sigma**dimension
        * mp.exp(-z * z / (4 * sigma**2))
        * (1 + dimension / (2 * sigma**2) - z * z / (4 * sigma**4))
    )


def derivative_inner(r, s, dimension=1, sigma=mp.mpf(1)):
    return (-1) ** r * mp.diff(
        lambda z: correlation(z, dimension, sigma), mp.mpf(0), r + s
    )


def zero_gap_constants(dimension=1, sigma=mp.mpf(1)):
    # The odd opening direction -G'''/6 projects only onto G' among
    # the tangent columns at -G': the other columns have incompatible parity.
    norm1 = derivative_inner(1, 1, dimension, sigma)
    norm3 = derivative_inner(3, 3, dimension, sigma)
    cross = derivative_inner(3, 1, dimension, sigma)
    c4 = (norm3 - cross * cross / norm1) / 72
    relative_error_over_t = mp.sqrt(2 * c4 / norm1)
    return {"c4": c4, "relative_error_over_t": relative_error_over_t}


def refitted_symmetric_pair(h):
    target_norm2 = 2 * correlation(0) + 2 * correlation(2)
    gram_diagonal = correlation(0)
    gram_offdiagonal = correlation(2 * h)
    cross = correlation(h - 1) + correlation(h + 1)
    weight = cross / (gram_diagonal + gram_offdiagonal)
    gap = target_norm2 / 2 - cross * weight
    return {
        "weight_each": weight,
        "energy_gap": gap,
        "relative_H1_error": mp.sqrt(max(mp.mpf(0), 2 * gap / target_norm2)),
    }


def main():
    results = {
        "arithmetic_digits": mp.mp.dps,
        "zero_gap": {str(d): zero_gap_constants(d) for d in (1, 2, 5, 10)},
        "upper_bound_example": {
            str(h): refitted_symmetric_pair(h)
            for h in (mp.mpf(".04"), mp.mpf(".6"), mp.mpf(1))
        },
    }
    c4 = results["zero_gap"]["1"]["c4"]
    assert abs(c4 - 13 * mp.sqrt(mp.pi) / 320) < mp.mpf("1e-70")
    results["upper_bound_example"]["dJ_dt_at_t_0.36"] = mp.diff(
        lambda t: refitted_symmetric_pair(mp.sqrt(t))["energy_gap"], mp.mpf(".36")
    )
    assert results["upper_bound_example"]["dJ_dt_at_t_0.36"] < 0
    assert results["upper_bound_example"]["1.0"]["relative_H1_error"] == 0

    # Dimension-one exception to the original c != 0 assertion:
    # v=G, u*=G+epsilon(G''+5G/6), theta=(a,c,beta)=(1,0,0).
    A = derivative_inner(0, 0)
    B = derivative_inner(1, 1)
    epsilon = mp.mpf(".1")
    residual_curvature = epsilon * (
        derivative_inner(2, 2) + mp.mpf(5) / 6 * derivative_inner(0, 2)
    )
    assert abs(residual_curvature - epsilon * 19 * mp.sqrt(mp.pi) / 12) < mp.mpf("1e-70")
    hessian = mp.matrix([
        [A, 0, 0],
        [0, B, B - residual_curvature],
        [0, B - residual_curvature, B - residual_curvature],
    ])
    eigenvalues = list(mp.eigsy(hessian, eigvals_only=True))
    assert min(eigenvalues) > 0
    results["one_dimensional_c_zero_exception"] = {
        "epsilon": epsilon,
        "hessian_eigenvalues": eigenvalues,
        "opening_slope": -residual_curvature / 2,
    }

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, mp.mpf):
            return mp.nstr(value, 40)
        return value

    output = Path(__file__).with_name("pair_analysis_checks.json")
    output.write_text(json.dumps(convert(results), indent=2) + "\n")
    print(f"All exact-formula checks passed; results: {output}")
    print("c4(d=1):", mp.nstr(c4, 18))
    print("E at h=.6:", mp.nstr(refitted_symmetric_pair(mp.mpf(".6"))["relative_H1_error"], 18))
    print("dJ/dt at t=.36:", mp.nstr(results["upper_bound_example"]["dJ_dt_at_t_0.36"], 18))


if __name__ == "__main__":
    main()
