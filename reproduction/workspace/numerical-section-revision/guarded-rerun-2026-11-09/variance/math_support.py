#!/usr/bin/env python3
"""Exact whole-space H1 formulas for the Gaussian-variance experiment.

The ordinary atoms are G(x-b)=exp(-(x-b)^2/2). The target is the
derivative of G_s(x)=exp(-x^2/(2s)), with s the variance, not its square
root. All functions return mpmath numbers at the caller's precision.

The centered completed reference is alpha_s G'. Its coefficient and error
are analytic. The positive inward derivative for s<1 establishes a local
boundary minimum near s=1; these formulas do not constitute a numerical
global-minimum certificate for a particular nonzero variance perturbation.
"""
from __future__ import annotations

import json

import mpmath as mp


def kernel(z):
    """(G(.−b),G(.−c))_H for z=b−c."""
    z = mp.mpf(z)
    return mp.sqrt(mp.pi) * mp.exp(-z*z/4) * (mp.mpf(3)/2-z*z/4)


def target_norm_squared(s):
    """||G_s'||_H^2, where H=H^1(R)."""
    s = mp.mpf(s)
    if s <= 0:
        raise ValueError("The target variance must be positive.")
    return mp.sqrt(mp.pi) * (2*s+3)/(4*s**mp.mpf("1.5"))


def cross_kernel(s, b):
    """(G_s,G(.−b))_H, before differentiating the target."""
    s, b = mp.mpf(s), mp.mpf(b)
    a = 1+s
    return mp.sqrt(2*mp.pi*s/a)*mp.exp(-b*b/(2*a))*(1+1/a-b*b/a**2)


def atom_load(s, b):
    """(G_s',G(.−b))_H = d/db cross_kernel(s,b)."""
    s, b = mp.mpf(s), mp.mpf(b)
    a = 1+s
    return mp.sqrt(2*mp.pi*s/a)*mp.exp(-b*b/(2*a))*(b**3/a**3-b*(1/a+3/a**2))


def atom_load_prime(s, b):
    """Derivative with respect to the model center b."""
    s, b = mp.mpf(s), mp.mpf(b)
    a = 1+s
    polynomial = -b**4/a**4+b*b*(1/a**2+6/a**3)-(1/a+3/a**2)
    return mp.sqrt(2*mp.pi*s/a)*mp.exp(-b*b/(2*a))*polynomial


def reference_coefficient(s):
    s = mp.mpf(s)
    return 4*mp.sqrt(2*s)*(s+4)/(5*(1+s)**mp.mpf("2.5"))


def reference_energy_gap(s):
    """0.5 ||G_s'−alpha_s G'||_H^2; numerical global optimality is separate."""
    s = mp.mpf(s)
    alpha = reference_coefficient(s)
    return (target_norm_squared(s)-alpha*alpha*5*mp.sqrt(mp.pi)/4)/2


def inward_slope(s):
    """d/dt of the fitted squared-error objective at beta=t=0, t=h^2."""
    s = mp.mpf(s)
    return 2*mp.sqrt(mp.pi)*s*(1-s)*(s+4)*(7*s+32)/(25*(1+s)**6)


def pair_gram_and_load(s, beta, h):
    """Diagonal Gram entries and target loads in the symmetric stable basis.

    E=(G(.−beta−h)+G(.−beta+h))/2,
    D=(G(.−beta−h)−G(.−beta+h))/(2h); D tends to −G'(.−beta).
    No numerical quadrature is used. Work at sufficient precision when h
    is small, because analytic Gram differences then contain cancellation.
    """
    s, beta, h = map(mp.mpf, (s, beta, h))
    if h < 0:
        raise ValueError("The half-gap must be nonnegative.")
    k0 = kernel(0)
    if h == 0:
        return k0, 5*mp.sqrt(mp.pi)/4, atom_load(s, beta), atom_load_prime(s, beta)
    k2 = kernel(2*h)
    plus, minus = atom_load(s, beta+h), atom_load(s, beta-h)
    return (k0+k2)/2, (k0-k2)/(2*h*h), (plus+minus)/2, (plus-minus)/(2*h)


def pair_evaluation(s, beta, h, a=None, c=None):
    """Evaluate aE+cD; omit both coefficients to fit them exactly."""
    s, beta, h = map(mp.mpf, (s, beta, h))
    ee, dd, re, rd = pair_gram_and_load(s, beta, h)
    if a is None and c is None:
        a, c = re/ee, rd/dd
    elif a is None or c is None:
        raise ValueError("Provide both coefficients or neither.")
    else:
        a, c = mp.mpf(a), mp.mpf(c)
    error2 = target_norm_squared(s)+a*a*ee+c*c*dd-2*(a*re+c*rd)
    gap = error2/2
    weight_norm = mp.sqrt((a*a+(c/h)**2)/2) if h > 0 else None
    return {"beta": beta, "half_gap": h, "a": a, "c": c,
            "energy_gap": gap, "relative_error": mp.sqrt(max(error2, 0)/target_norm_squared(s)),
            "excess_above_centered_reference": gap-reference_energy_gap(s),
            "ordinary_weight_norm": weight_norm,
            "stable_coefficient_norm": mp.sqrt(a*a+c*c)}


def reference_summary(s):
    s = mp.mpf(s)
    norm2 = target_norm_squared(s)
    gap = reference_energy_gap(s)
    return {"variance": s, "alpha": reference_coefficient(s),
            "target_norm_squared": norm2, "energy_gap": gap,
            "relative_error": mp.sqrt(max(2*gap, 0)/norm2),
            "inward_slope": inward_slope(s)}


if __name__ == "__main__":
    mp.mp.dps = 80
    print(json.dumps([{k: mp.nstr(v, 40) for k, v in reference_summary(s).items()}
                      for s in ("0.99", "0.995", "0.999")], indent=2))
