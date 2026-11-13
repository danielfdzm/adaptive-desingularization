#!/usr/bin/env python3
"""Audit saved endpoints independently at 80 digits and changed quadrature.

This is a validation of the small benchmark, not of the historical cohorts.
High-precision H1 errors expand finite pairs into ordinary translates and
differentiate the ordinary correlation for completed columns. They do not
reuse the float64 hyperbolic parity formula.
"""
import json
from pathlib import Path
import mpmath as mp
import numpy as np
import run

mp.mp.dps = 80


def m(x):
    return mp.mpf(str(float(x)))


def primitives(z, a):
    result = []
    for row, coeff in zip(z, np.asarray(a).reshape(-1, 2)):
        c, t = list(map(m, row[:2])), m(row[2])
        e = list(map(m, row[3:5]))
        norm = mp.sqrt(sum(x*x for x in e))
        e = [x/norm for x in e]
        aa, cc = map(m, coeff)
        if t == 0:
            result.extend([(c, e, 0, aa), (c, e, 1, cc)])
        else:
            h = mp.sqrt(t)
            for sign in (1, -1):
                result.append(([ci+sign*h*ei for ci,ei in zip(c,e)], e, 0, aa/2+sign*cc/(2*h)))
    return result


def correlation(pa, pb):
    ca, ea, oa, _ = pa
    cb, eb, ob, _ = pb
    def shifted(s, t):
        rr = sum((a+s*e-b-t*f)**2 for a,e,b,f in zip(ca,ea,cb,eb))
        return mp.pi*mp.exp(-rr/4)*(2-rr/4)
    return mp.diff(shifted, (mp.mpf(0), mp.mpf(0)), (oa, ob))


def squared_norm(atoms):
    return mp.fsum(a[3]*b[3]*correlation(a,b) for a in atoms for b in atoms)


def main():
    folder = Path(__file__).parent
    data = json.loads((folder/"results.json").read_text())
    target = primitives(run.TARGET_Z, run.TARGET_A)
    target_norm = squared_norm(target)
    checks = []
    for record in data["runs"]:
        endpoint = record["trace"][-1]
        z, a = np.array(endpoint["z"]), np.array(endpoint["a"])
        error_atoms = primitives(z, a) + [(c,e,o,-w) for c,e,o,w in target]
        relative = mp.sqrt(squared_norm(error_atoms)/target_norm)
        assert abs(float(relative)-record["relative_H1_error"]) < 2e-9
        accepted = [row for row in record["trace"] if "accepted_step" in row]
        assert all(row["descent_margin"] >= 0 for row in accepted)
        assert record["max_scaled_coefficient_residual"] <= run.COEFFICIENT_TOL
        quadchecks = []
        for order in (48, 64):
            problem = run.Problem(order)
            value = problem.evaluate(z, a)
            quadchecks.append({"order": order,
                "canonical_residual_after_refit": run.canonical_residual(z, value["gradient"]),
                "refitted_coefficient_change": float(np.linalg.norm(value["a"]-a)),
                "refitted_objective": value["F"]})
        checks.append({"seed": record["seed"], "method": record["method"],
            "H1_error_80_digits": mp.nstr(relative, 45),
            "float64_H1_error_absolute_difference": abs(float(relative)-record["relative_H1_error"]),
            "minimum_accepted_descent_margin": min(row["descent_margin"] for row in accepted),
            "minimum_accepted_step": min(row["accepted_step"] for row in accepted),
            "maximum_saved_objective_increase": max(0., max(b["objective"]-a["objective"]
                for a,b in zip(record["trace"][:-1], record["trace"][1:]))),
            "sphere_norm_max_error": max(abs(np.linalg.norm(np.array(row["z"])[:, 3:5], axis=1)-1).max() for row in record["trace"]),
            "quadrature_checks": quadchecks})
    payload = {"precision_digits": 80, "H1_validation": "independent ordinary/confluent Gaussian correlation expansion",
        "quadrature_interpretation": "endpoint sensitivity to 48x48 and64x64 nonlinear quadrature, with each forcing manufactured consistently; not a continuous-energy certificate",
        "runs": checks}
    (folder/"validation.json").write_text(json.dumps(payload, indent=2)+"\n")
    for row in checks:
        print(row["seed"], row["method"], "80-digit error", row["H1_error_80_digits"],
              "R64", row["quadrature_checks"][-1]["canonical_residual_after_refit"])


if __name__ == "__main__":
    main()
