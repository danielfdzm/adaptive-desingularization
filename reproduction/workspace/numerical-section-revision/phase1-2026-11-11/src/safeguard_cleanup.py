"""Frozen-chart safeguarded projected-gradient cleanup (Phase 2, 11 Nov 2026).

This is the final phase behind Theorem thm:stationarity of the paper ("Stationarity of a
safeguarded version", Section 4.3).  After solver.run() has stopped, the chart is frozen: the
block structure, the tangent frames of the directions and the L-BFGS-B box are fixed at entry.
The objective is the solver's normalized training objective with coefficients refitted at every
geometry (variable projection), i.e. exactly what solver.evaluate returns.

Each iteration computes the projected-gradient residual, with gamma = 1 and unit scale,
    R = (x - P(x - gamma * g)) / gamma,      P = projection onto the box,
whose max-abs norm is the L-BFGS-B pgtol measure.  If max|R| > tol, it takes the Armijo step on
the segment x + s d, d = -gamma R, s = 1, beta, beta^2, ..., accepting the first trial with
    f(x + s d) <= f(x) - eta * s * |d|^2 / gamma .
Accepted steps are committed with solver.accept_cached(), so on return the solver holds the
cleanup endpoint.  The main endpoint of the run must be recorded before calling cleanup().

Exits (field "stop"):
    tolerance     max|R| <= tol: stationary to tol in the frozen chart (the theorem's exit).
    rounding      the decrease predicted for the next trial, s |d|^2 / gamma, is below the
                  verification floor: rounding_factor * eps * max(1, |f|), or noise_factor times the
                  measured noise of the refitted objective (warm versus cold coefficient start at the
                  entry point), whichever is larger.  No further descent can be verified there.
                  Accepted steps must decrease the objective by more than this floor as well as by the
                  Armijo amount.
    inadmissible  every trial along d is rejected by the directional guard, is nonfinite, or has a
                  failed coefficient solve: the iterate is not a stationary point of the box problem.
    no_decrease   backtracking exhausted without a verified decrease.
    work_limit    max_evals or the solver's fit budget (cs.WorkLimit) was reached.
    fit_failure   the coefficient solve failed at the entry point (FloatingPointError).
    inadmissible_start  the frozen chart cannot be evaluated at entry.
A work-limit, inadmissible or fit-failure exit is not a stationarity exit.

As in the solver's amendment of 11 Nov 2026 (Solver.evaluate_trial), a failed coefficient solve at a
trial point makes that trial inadmissible: the backtracking continues, and the count is reported in
"trial_fit_failures".  A failure of the cold-start noise probe leaves the noise estimate at zero and
sets "noise_probe_failed".

Usage (Phase 1 chain solver):
    import safeguard_cleanup as sc
    record = solver.run()                      # main endpoint, recorded as usual
    solver.max_fits = solver.fits + max_evals  # or raise it by max_evals
    record["cleanup"] = sc.cleanup(solver, tol=1e-7, max_evals=200)

Each objective evaluation is one coefficient fit, counted by the solver as usual.
"""
from __future__ import annotations

import math

import numpy as np

import chain_solver as cs

EPS = np.finfo(float).eps


def _residual(x, g, lo, hi, gamma):
    return (x - np.clip(x - gamma * g, lo, hi)) / gamma


def _max_abs(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


def _evaluate_trial(solver, x, geo):
    """solver.evaluate, with a failed coefficient solve treated as an inadmissible trial (+inf)."""
    try:
        return solver.evaluate(x, geo), False
    except FloatingPointError:
        return (float("inf"), None), True


def cleanup(solver, tol=1e-7, max_evals=200, gamma=1.0, eta=1e-4, beta=0.5, max_backtracks=50,
            rounding_factor=16.0, noise_factor=3.0, trace_limit=200, high_precision=False):
    """Safeguarded projected-gradient cleanup in the frozen final chart; returns a JSON-ready dict.

    The verification floor is max(rounding_factor * eps * max(1, |f|), noise_factor * noise), where
    noise = |f_cold - f_warm| compares the refitted objective at the entry point computed from the
    warm and from a cold coefficient start (one extra fit, only if the entry point misses tol).  Below
    this floor a function-value test cannot certify a decrease, and the cleanup stops with "rounding"."""
    out = {"tol": tol, "gamma": gamma, "eta": eta, "beta": beta, "max_evals": max_evals,
           "norm": "max-abs projected gradient, unit scale, w.r.t. the L-BFGS-B box (pgtol measure)",
           "objective": "normalized training objective with refitted coefficients (solver.evaluate)",
           "chart": "frozen at entry: blocks, tangent frames and bounds of cs.Geometry(scale=1)"}
    evals, accepted, trace, trial_fit_failures = 0, 0, [], 0
    try:
        geo = cs.Geometry(solver.blocks, solver.d, solver.sigma, solver.box)
    except Exception as exc:                                   # the final chart itself is invalid
        out.update(stop="inadmissible_start", error=repr(exc), evaluations=0, iterations=0)
        return out
    lo = np.array([a for a, _ in geo.bounds], dtype=float)
    hi = np.array([b for _, b in geo.bounds], dtype=float)
    x = geo.x.astype(float).copy()
    x0 = x.copy()
    solver.cache = None
    solver.warm = None if solver.coef is None else solver.coef.copy()
    stop, final = None, None
    out["initial"] = None
    try:
        f, g = solver.evaluate(x, geo)
        evals += 1
        if not np.isfinite(f):
            out.update(stop="inadmissible_start", evaluations=evals, iterations=0)
            return out
        solver.accept_cached()
        R = _residual(x, g, lo, hi, gamma)
        pg = _max_abs(R)
        err = solver.rel_error()
        out["initial"] = {"objective": f, "pg_max": pg, "rel_error_double": err}
        final = {"objective": f, "pg_max": pg, "rel_error_double": err}
        noise = 0.0
        out["noise_probe_failed"] = False
        if pg > tol and evals < max_evals:                    # objective noise of the refit at entry
            saved_cache, saved_warm = solver.cache, solver.warm
            solver.cache, solver.warm = None, None
            (fc, _), failed = _evaluate_trial(solver, x, geo)
            evals += 1
            if np.isfinite(fc):
                noise = abs(fc - f)
            out["noise_probe_failed"] = bool(failed)
            solver.cache, solver.warm = saved_cache, saved_warm
            solver.accept_cached()
        out["objective_noise"] = noise
        k = 0
        while True:
            if len(trace) < trace_limit:
                trace.append({"iteration": k, "objective": f, "pg_max": pg, "evaluations": evals})
            if pg <= tol:
                stop = "tolerance"
                break
            if evals >= max_evals:
                stop = "work_limit"
                break
            d = -gamma * R
            dd = float(d @ d)
            floor = max(rounding_factor * EPS * max(1.0, abs(f)), noise_factor * noise)
            s, step_ok, reason = 1.0, False, "no_decrease"
            any_finite, n_trials = False, 0
            for _ in range(max_backtracks):
                # <g, d> <= -|d|^2/gamma, so for small s the decrease is about s |d|^2 / gamma.  Below the
                # rounding floor no decrease can be verified; accepted decreases must also exceed the floor.
                predicted = s * dd / gamma
                if predicted <= floor:
                    reason = "rounding"
                    break
                if evals >= max_evals:
                    reason = "work_limit"
                    break
                xt = np.clip(x + s * d, lo, hi)               # stays in the box; clip guards rounding only
                (ft, gt), failed = _evaluate_trial(solver, xt, geo)
                trial_fit_failures += int(failed)
                evals += 1
                n_trials += 1
                if np.isfinite(ft):
                    any_finite = True
                    if ft <= f - max(eta * predicted, floor):
                        step_ok = True
                        break
                s *= beta
            if not step_ok:
                if n_trials > 0 and not any_finite and reason != "work_limit":
                    reason = "inadmissible"
                stop = reason
                break
            solver.accept_cached()
            accepted += 1
            x, f, g = xt, ft, gt
            R = _residual(x, g, lo, hi, gamma)
            pg = _max_abs(R)
            final = {"objective": f, "pg_max": pg, "rel_error_double": solver.rel_error()}
            k += 1
    except cs.WorkLimit:
        stop = "work_limit"
    except FloatingPointError as exc:
        stop = "fit_failure"
        out["error"] = repr(exc)
    solver.cache = None                                        # never leave a rejected trial in the cache
    change = None
    if final is not None and out["initial"] is not None:
        change = final["objective"] - out["initial"]["objective"]
    out.update(stop=stop, stationary=(stop == "tolerance"), iterations=accepted, evaluations=evals,
               trial_fit_failures=trial_fit_failures, final=final, objective_change=change,
               max_abs_parameter_change=_max_abs(x - x0), trace=trace)
    if high_precision and final is not None:
        A = solver.state_primitives()
        B = [(c, u, o, -w) for c, u, o, w in solver.target_np]
        num = cs.ck.mp_norm2(A + B, solver.sigma, solver.d, dps=80)
        den = cs.ck.mp_norm2(solver.target_np, solver.sigma, solver.d, dps=80)
        out["final"]["rel_error_80_digits"] = float((max(num, 0) / den) ** 0.5)
    return out
