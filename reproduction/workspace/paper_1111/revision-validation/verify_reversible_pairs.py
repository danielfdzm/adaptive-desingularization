#!/usr/bin/env python3
"""Small reference implementation of safeguarded reversible pair optimization.

One-dimensional linear H1 problems, exact Gaussian correlations, and
70-digit arithmetic isolate the proposed algorithm from quadrature and
double-precision objective cancellation. Geometry is always (beta,t);
storage changes between stable and ordinary pair coordinates by hysteresis.
The full projected-gradient trial is used, not an interpolated trial.
In ordinary mode the persistent data are the two centers and weights;
the inverse map reconstructs canonical variables before every iteration.
Descent remains canonical (equivalently, uses the pulled-back geometry);
this is not Euclidean gradient descent in raw centers and weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 70
SQRT_PI = mp.sqrt(mp.pi)


def kernel(q, variance=mp.mpf(1)):
    total = 1 + variance
    return mp.sqrt(2 * mp.pi * variance / total) * mp.exp(-q*q/(2*total)) * (
        1 + 1/total - q*q/total**2
    )


def pair_correlations(q, t, variance=mp.mpf(1)):
    """Inner products of M,D with a Gaussian of the given variance.

    cosh and sinh-over-gap are entire in t. The continuation to negative
    t permits analytic numerical differentiation at t=0.
    """
    total = 1 + variance
    root = mp.sqrt(t)
    z = q * root / total
    cosh = mp.cosh(z)
    sinc = q / total if t == 0 else mp.sinh(z) / root
    base = mp.sqrt(2 * mp.pi * variance / total) * mp.exp(-(q*q+t)/(2*total))
    poly = 1 + 1/total - (q*q+t)/total**2
    M = base * (poly*cosh + 2*q*t*sinc/total**2)
    D = base * (-poly*sinc - 2*q*cosh/total**2)
    return mp.re(M), mp.re(D)


def gram(t):
    ex = mp.exp(-t)
    MM = SQRT_PI * (mp.mpf("1.5") + ex*(mp.mpf("1.5")-t)) / 2
    DD = SQRT_PI * mp.mpf("1.25") if t == 0 else SQRT_PI * (
        -mp.mpf("1.5")*mp.expm1(-t) + t*ex
    ) / (2*t)
    return MM, DD


class Problem:
    def __init__(self, name, kind, variance=mp.mpf(1)):
        self.name, self.kind, self.variance = name, kind, variance
        if kind == "derivative":
            s = variance
            # ||G_s'||_H^2 from exact derivative integrals.
            self.norm2 = mp.sqrt(mp.pi) * (1/(2*mp.sqrt(s)) + 3/(4*s**mp.mpf("1.5")))
        else:
            self.norm2 = 2*kernel(0) + 2*kernel(2)

    def rhs(self, beta, t):
        if self.kind == "derivative":
            return tuple(mp.diff(lambda q: pair_correlations(q, t, self.variance)[j], beta)
                         for j in (0, 1))
        left = pair_correlations(beta-1, t)
        right = pair_correlations(beta+1, t)
        return left[0]+right[0], left[1]+right[1]

    def evaluate(self, beta, t, with_gradient=True):
        MM, DD = gram(t)
        bM, bD = self.rhs(beta, t)
        a, c = bM/MM, bD/DD
        F = (self.norm2 - bM*a - bD*c) / (2*self.norm2)
        out = {"F": F, "a": a, "c": c, "relative_error": mp.sqrt(max(0, 2*F))}
        if with_gradient:
            bMb, bDb = (mp.diff(lambda q: self.rhs(q, t)[j], beta) for j in (0, 1))
            bMt, bDt = (mp.diff(lambda x: self.rhs(beta, x)[j], t) for j in (0, 1))
            MMt, DDt = (mp.diff(lambda x: gram(x)[j], t) for j in (0, 1))
            out["gradient"] = (
                -(a*bMb+c*bDb)/self.norm2,
                (a*a*MMt/2+c*c*DDt/2-a*bMt-c*bDt)/self.norm2,
            )
        return out


def project(beta, t):
    return min(mp.mpf(3), max(mp.mpf(-3), beta)), min(mp.mpf(4), max(mp.mpf(0), t))


def residual(beta, t, gradient):
    trial = project(beta-gradient[0], t-gradient[1])
    return mp.sqrt((beta-trial[0])**2+(t-trial[1])**2)


def store_pair(beta, t, a, c, mode):
    if mode == "paired":
        return {"mode": mode, "beta": beta, "t": t, "a": a, "c": c}
    h = mp.sqrt(t)
    assert h > 0
    return {"mode": mode, "bplus": beta+h, "bminus": beta-h,
            "wplus": a/2+c/(2*h), "wminus": a/2-c/(2*h)}


def canonical_from_storage(storage):
    if storage["mode"] == "paired":
        return tuple(storage[key] for key in ("beta", "t", "a", "c"))
    beta = (storage["bplus"]+storage["bminus"])/2
    h = (storage["bplus"]-storage["bminus"])/2
    assert h > 0
    return beta, h*h, storage["wplus"]+storage["wminus"], h*(
        storage["wplus"]-storage["wminus"])


def solve(problem, initial, tolerance=mp.mpf("1e-12"), limit=1200):
    beta, t = map(mp.mpf, initial)
    on, off = mp.mpf(".4")**2, mp.mpf(".6")**2
    mode = "ordinary" if t >= off else "paired"
    initial_fit = problem.evaluate(beta, t, with_gradient=False)
    storage = store_pair(beta, t, initial_fit["a"], initial_fit["c"], mode)
    eta, shrink, alpha0 = mp.mpf(".5"), mp.mpf(".5"), mp.mpf(1)
    trace, events, failures = [], [], []
    prior_objective = None
    first_zero = None
    for k in range(limit+1):
        beta, t, stored_a, stored_c = canonical_from_storage(storage)
        mode = storage["mode"]
        values = problem.evaluate(beta, t)
        assert abs(values["a"]-stored_a)+abs(values["c"]-stored_c) < mp.mpf("1e-55")
        R = residual(beta, t, values["gradient"])
        trace.append({"iteration": k, "beta": beta, "t": t, "mode": mode,
                      "objective": values["F"], "residual": R,
                      "relative_target_error": values["relative_error"],
                      "opening_slope_normalized": values["gradient"][1],
                      "opening_slope_physical": values["gradient"][1]*problem.norm2})
        if prior_objective is not None:
            assert values["F"] <= prior_objective + mp.mpf("1e-55")
        if t == 0 and first_zero is None:
            first_zero = k
        if R <= tolerance:
            break
        alpha = alpha0
        while True:
            proposal = project(beta-alpha*values["gradient"][0], t-alpha*values["gradient"][1])
            distance2 = (proposal[0]-beta)**2+(proposal[1]-t)**2
            candidate = problem.evaluate(*proposal, with_gradient=False)
            if candidate["F"] <= values["F"]-eta*distance2/(2*alpha):
                break
            alpha *= shrink
            if alpha < mp.mpf("1e-40"):
                failures.append("backtracking stalled")
                break
        if failures:
            break
        prior_objective = values["F"]
        beta, t = proposal
        new_mode = ("ordinary" if t >= off else "paired") if mode == "paired" else (
            "paired" if t <= on else "ordinary")
        storage = store_pair(beta, t, candidate["a"], candidate["c"], new_mode)
        restored = canonical_from_storage(storage)
        coordinate_error = sum(abs(x-y) for x,y in zip(
            restored, (beta, t, candidate["a"], candidate["c"])))
        assert coordinate_error < mp.mpf("1e-60")
        if new_mode != mode:
            events.append({"iteration": k+1, "from": mode, "to": new_mode,
                           "t": t, "beta": beta, "roundtrip_coordinate_error":
                           coordinate_error})
    return {"problem": problem.name, "initial": initial, "iterations": trace[-1]["iteration"],
            "first_zero_iteration": first_zero, "events": events, "failures": failures,
            "final": trace[-1], "final_storage": storage, "trace": trace}


def main():
    problems = [
        (Problem("zero_gap_derivative", "derivative"), (".02", ".04")),
        (Problem("positive_gap_derivative_s099", "derivative", mp.mpf(".99")), (".02", ".04")),
        (Problem("reopen_to_exact_ordinary_pair", "ordinary"), ("0", ".0016")),
        (Problem("ordinary_to_completed_pair_s099", "derivative", mp.mpf(".99")), ("0", ".64")),
        (Problem("negative_slope_reopens_from_t0", "ordinary"), ("0", "0")),
    ]
    results = [solve(problem, initial) for problem, initial in problems]
    assert all(not r["failures"] for r in results)
    assert all(r["final"]["residual"] <= mp.mpf("1e-12") for r in results)
    assert results[1]["first_zero_iteration"] is not None
    assert results[1]["final"]["t"] == 0
    assert results[2]["events"] and results[2]["events"][0]["to"] == "ordinary"
    assert abs(results[2]["final"]["t"]-1) < mp.mpf("1e-10")
    assert results[2]["final"]["relative_target_error"] < mp.mpf("1e-10")
    assert results[3]["events"] and results[3]["events"][0]["to"] == "paired"
    assert results[3]["final"]["t"] == 0
    assert results[4]["trace"][1]["t"] > 0
    assert results[4]["events"] and results[4]["events"][0]["to"] == "ordinary"
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, mp.mpf):
            return mp.nstr(value, 35)
        return value
    output = Path(__file__).with_name("reversible_pair_checks.json")
    parameters = {
        "objective": "0.5 * ||V-u_target||_H1^2 / ||u_target||_H1^2",
        "coefficient_fits": "exact 2x2 diagonal Gram solve at every trial, in 70-digit arithmetic",
        "ordinary_storage": "persistent (bplus,bminus,wplus,wminus), inverted before every iteration",
        "geometry_metric": "canonical (beta,t); ordinary storage uses exact pullback, not raw Euclidean descent",
        "beta_interval": [-3, 3],
        "t_interval": [0, 4],
        "h_on": 0.4,
        "h_off": 0.6,
        "alpha0": 1,
        "backtracking_shrink": 0.5,
        "eta": 0.5,
        "descent_test": "F(trial) <= F(current) - eta * ||trial-current||^2 / (2*alpha)",
        "residual_step": 1,
        "residual_norm": "Euclidean",
        "residual_tolerance": "1e-12",
        "maximum_steps": 1200,
        "precision_digits": mp.mp.dps,
        "quadrature": "none; analytic whole-space Gaussian H1 correlations",
        "gradient": "mpmath differentiation of analytic correlation formulas",
    }
    output.write_text(json.dumps({"parameters": parameters, "results": convert(results)}, indent=2)+"\n")
    for r in results:
        print(r["problem"], "steps", r["iterations"], "first t=0", r["first_zero_iteration"],
              "switches", len(r["events"]), "R", mp.nstr(r["final"]["residual"], 5),
              "E", mp.nstr(r["final"]["relative_target_error"], 5))
    summary = ["problem,iterations,switches,first_zero_iteration,residual,relative_target_error,"
               "initial_opening_slope_physical,final_opening_slope_physical"]
    for r in results:
        summary.append(",".join([
            r["problem"], str(r["iterations"]), str(len(r["events"])),
            "" if r["first_zero_iteration"] is None else str(r["first_zero_iteration"]),
            mp.nstr(r["final"]["residual"], 12),
            mp.nstr(r["final"]["relative_target_error"], 12),
            mp.nstr(r["trace"][0]["opening_slope_physical"], 12),
            mp.nstr(r["final"]["opening_slope_physical"], 12)]))
    Path(__file__).with_name("reversible_pair_summary.csv").write_text("\n".join(summary)+"\n")
    print("All projected-descent, exact-transition, reopening, and identification checks passed.")


if __name__ == "__main__":
    main()
