#!/usr/bin/env python3
"""Small float64 check of Algorithm 4.2: two rotating pairs, nonlinear PB.

The H1 quadratic term and forcing correlations use closed Gaussian formulas.
Only the nonlinear term uses fixed positive tensor Gauss--Legendre quadrature.
This is an inexact-arithmetic experiment, not a verification of exact iterates.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
DT = torch.float64
M, D, SIGMA, LAMBDA = 2, 2, 1.0, 5.0
CENTERS = np.array([[-1.5, 0.0], [1.5, 0.0]])
LOW = np.column_stack((CENTERS - .25, np.zeros(2)))
HIGH = np.column_stack((CENTERS + .25, np.full(2, .36)))
TARGET_ANGLES = np.array([.4, 1.1])
TARGET_Z = np.column_stack((CENTERS, [0., .16], np.cos(TARGET_ANGLES), np.sin(TARGET_ANGLES)))
TARGET_A = np.array([.5, .7, -.35, -.65])
TAU_IN, TAU_OUT = .15**2, .25**2
COEFFICIENT_TOL = 2e-13


def tensor(x):
    return torch.as_tensor(x, dtype=DT)


def cosh_root(q):
    """cosh(sqrt(q)), with a differentiable analytic branch at zero."""
    small = q < 1e-5
    safe = torch.where(small, torch.ones_like(q), q)
    poly = 1 + q*(.5 + q*(1/24 + q*(1/720 + q*(1/40320 + q/3628800))))
    return torch.where(small, poly, torch.cosh(torch.sqrt(safe)))


def sinhc_root(q):
    """sinh(sqrt(q))/sqrt(q), including the analytic value at zero."""
    small = q < 1e-5
    safe = torch.where(small, torch.ones_like(q), q)
    poly = 1 + q*(1/6 + q*(1/120 + q*(1/5040 + q*(1/362880 + q/39916800))))
    return torch.where(small, poly, torch.sinh(torch.sqrt(safe))/torch.sqrt(safe))


def parts(z):
    e = z[:, 3:5]
    e = e / torch.linalg.vector_norm(e, dim=1, keepdim=True)
    return z[:, :2], z[:, 2], e


def gram(za, zb):
    """Exact analytic H1(R2) Gram in stable M,D coordinates, sigma=1.

    Four parity sums of the Gaussian correlation are expressed in cosh
    and sinhc. Every formula is analytic in both squared half-gaps at zero.
    The tiny-argument series above approximate entire scalar functions only.
    """
    ca, ta, ea = parts(za)
    cb, tb, eb = parts(zb)
    r = ca[:, None, :] - cb[None, :, :]
    t, u = ta[:, None], tb[None, :]
    x = -(r*ea[:, None, :]).sum(-1)/2
    y = (r*eb[None, :, :]).sum(-1)/2
    z = (ea[:, None, :]*eb[None, :, :]).sum(-1)/2
    cx, cy, cz = cosh_root(t*x*x), cosh_root(u*y*y), cosh_root(t*u*z*z)
    sx, sy, sz = x*sinhc_root(t*x*x), y*sinhc_root(u*y*y), z*sinhc_root(t*u*z*z)
    c00 = cx*cy*cz + t*u*sx*sy*sz
    c10 = sx*cy*cz + u*cx*sy*sz
    c01 = cx*sy*cz + t*sx*cy*sz
    c11 = sx*sy*cz + cx*cy*sz
    rr = r.square().sum(-1) + t + u
    p = 1 + D/2 - rr/4
    factor = np.pi**(D/2)*torch.exp(-rr/4)
    mm = factor*(p*c00 + t*x*c10 + u*y*c01 + t*u*z*c11)
    dm = factor*(p*c10 + x*c00 + u*y*c11 + u*z*c01)
    md = factor*(p*c01 + t*x*c11 + y*c00 + t*z*c10)
    dd = factor*(p*c11 + x*c01 + y*c10 + z*c00)
    # (pair a, column a, pair b, column b)
    return torch.stack((torch.stack((mm, md), -1), torch.stack((dm, dd), -1)), 1).reshape(2*len(za), 2*len(zb))


def values(z, points):
    c, t, e = parts(z)
    r = points[:, None, :] - c[None, :, :]
    q = (r*e[None, :, :]).sum(-1)
    factor = torch.exp(-(r.square().sum(-1)+t[None, :])/2)
    return torch.stack((factor*cosh_root(t*q*q), factor*q*sinhc_root(t*q*q)), -1).reshape(len(points), 2*len(z))


def nonlinear_remainders(delta):
    small = delta.abs() < .1
    q = delta*delta
    r3series = delta*q*(1/6+q*(1/120+q*(1/5040+q*(1/362880+q/39916800))))
    r4series = q*q*(1/24+q*(1/720+q*(1/40320+q*(1/3628800+q/479001600))))
    r3 = torch.where(small, r3series, torch.sinh(delta)-delta)
    r4 = torch.where(small, r4series, 2*torch.sinh(delta/2)**2-q/2)
    return r3, r4


class Problem:
    def __init__(self, order=32):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        xx, yy = np.meshgrid(7*nodes, 7*nodes, indexing="ij")
        self.points = tensor(np.column_stack((xx.ravel(), yy.ravel())))
        self.weights = tensor((49*weights[:, None]*weights[None, :]).ravel())
        self.target_z, self.target_a = tensor(TARGET_Z), tensor(TARGET_A)
        self.target_gram = gram(self.target_z, self.target_z)
        self.norm2 = self.target_a @ self.target_gram @ self.target_a
        self.target = values(self.target_z, self.points) @ self.target_a
        self.cosh_u, self.sinh_u = torch.cosh(self.target), torch.sinh(self.target)
        self.cosh_u_minus_one = 2*torch.sinh(self.target/2)**2
        self.fits = self.coefficient_steps = 0
        self.max_fit_residual = self.max_fit_scaled_residual = 0.

    def objective(self, a, G, cross, phi):
        delta = phi @ a - self.target
        r3, r4 = nonlinear_remainders(delta)
        n = .5*self.cosh_u_minus_one*delta**2 + self.cosh_u*r4 + self.sinh_u*r3
        error2 = a @ G @ a - 2*a @ cross + self.norm2
        return (.5*error2 + LAMBDA*(self.weights*n).sum())/self.norm2

    def coefficient_fit(self, G, cross, phi, initial=None):
        self.fits += 1
        a = torch.linalg.solve(G, cross) if initial is None else tensor(initial).clone()
        for it in range(30):
            v = phi@a
            delta = v-self.target
            r3, _ = nonlinear_remainders(delta)
            residual = self.cosh_u_minus_one*delta + self.cosh_u*r3 + self.sinh_u*(2*torch.sinh(delta/2)**2)
            g = G@a-cross+LAMBDA*phi.T@(self.weights*residual)
            scaled = torch.linalg.vector_norm(g)/(1+torch.linalg.vector_norm(cross))
            if float(scaled) <= COEFFICIENT_TOL:
                break
            H = G+LAMBDA*phi.T@((self.weights*(2*torch.sinh(v/2)**2))[:, None]*phi)
            step = torch.linalg.solve(H, g)
            F = self.objective(a, G, cross, phi)
            alpha = 1.
            # Full Newton is accepted near the minimizer even when its tiny
            # decrease is below objective precision; the residual controls fit.
            while alpha > 2**-20:
                candidate = a-alpha*step
                value = self.objective(candidate, G, cross, phi)
                if float(value) <= float(F) - 1e-4*alpha*float(g@step)/float(self.norm2) + 8e-16:
                    break
                alpha *= .5
            a = candidate
            self.coefficient_steps += 1
        else:
            raise RuntimeError("coefficient Newton fit did not converge")
        raw = float(torch.linalg.vector_norm(g))
        scaled = float(scaled)
        self.max_fit_residual = max(self.max_fit_residual, raw)
        self.max_fit_scaled_residual = max(self.max_fit_scaled_residual, scaled)
        return a

    def evaluate(self, z_np, initial=None, need_gradient=True):
        z = tensor(z_np).clone().requires_grad_(need_gradient)
        G, cross, phi = gram(z, z), gram(z, self.target_z)@self.target_a, values(z, self.points)
        a = self.coefficient_fit(G.detach(), cross.detach(), phi.detach(), initial)
        F = self.objective(a, G, cross, phi)
        g = torch.autograd.grad(F, z)[0].detach().numpy() if need_gradient else None
        err2 = float((a @ G @ a - 2*a @ cross + self.norm2).detach())
        return {"F": float(F.detach()), "a": a.numpy(), "gradient": g,
                "relative_H1_error": np.sqrt(max(0., err2/float(self.norm2)))}


def project(z, gradient, alpha, coordinates):
    out = z.copy()
    out[:, :2] = np.clip(z[:, :2]-alpha*gradient[:, :2], LOW[:, :2], HIGH[:, :2])
    if coordinates == "t":
        out[:, 2] = np.clip(z[:, 2]-alpha*gradient[:, 2], 0, .36)
    else:
        h = np.sqrt(z[:, 2])
        out[:, 2] = np.clip(h-alpha*2*h*gradient[:, 2], 0, .6)**2
    out[:, 3:5] = z[:, 3:5]-alpha*gradient[:, 3:5]
    out[:, 3:5] /= np.linalg.norm(out[:, 3:5], axis=1)[:, None]
    return out


def step_norm2(z1, z2, coordinates):
    diff = z2-z1
    if coordinates == "h":
        diff[:, 2] = np.sqrt(z2[:, 2])-np.sqrt(z1[:, 2])
    return float(np.sum(diff*diff))


def canonical_residual(z, g):
    dx = z[:, :3]-np.clip(z[:, :3]-g[:, :3], LOW, HIGH)
    return float(np.sqrt(np.sum(dx*dx)+np.sum(g[:, 3:5]**2)))


def storage_pack(z, a, modes):
    records = []
    for j, mode in enumerate(modes):
        if mode == "stable":
            records.append({"mode": mode, "z": z[j].tolist(), "a": a[2*j:2*j+2].tolist()})
        else:
            c, t, e = z[j, :2], z[j, 2], z[j, 3:5]
            h = np.sqrt(t)
            records.append({"mode": mode, "bplus": (c+h*e).tolist(), "bminus": (c-h*e).tolist(),
                            "wplus": float(a[2*j]/2+a[2*j+1]/(2*h)),
                            "wminus": float(a[2*j]/2-a[2*j+1]/(2*h))})
    return records


def storage_unpack(records):
    zs, coeffs = [], []
    for r in records:
        if r["mode"] == "stable":
            zs.append(r["z"])
            coeffs.extend(r["a"])
        else:
            plus, minus = np.array(r["bplus"]), np.array(r["bminus"])
            delta = (plus-minus)/2
            h = np.linalg.norm(delta)
            zs.append(np.r_[(plus+minus)/2, h*h, delta/h])
            coeffs.extend([r["wplus"]+r["wminus"], h*(r["wplus"]-r["wminus"])])
    return np.array(zs), np.array(coeffs)


def initial_geometry(seed):
    rng = np.random.default_rng(seed)
    z = TARGET_Z.copy()
    z[:, :2] += rng.uniform(-.08, .08, (2, 2))
    angles = TARGET_ANGLES + rng.uniform(-.2, .2, 2)
    z[:, 3:5] = np.column_stack((np.cos(angles), np.sin(angles)))
    z[:, 2] = {4101: [.12, .01], 4102: [.04, .04], 4103: [.20, .25]}[seed]
    return z


def solve(seed, coordinates, order, max_steps, fit_budget, tolerance):
    problem = Problem(order)
    z = initial_geometry(seed)
    start = time.perf_counter()
    result = problem.evaluate(z)
    modes = ["ordinary" if t >= TAU_OUT else "stable" for t in z[:, 2]]
    storage = storage_pack(z, result["a"], modes)
    trace, events = [], []
    status, max_roundtrip, worst_descent_violation = "step_limit", 0., 0.
    for k in range(max_steps+1):
        if coordinates == "t":
            z0, a0 = storage_unpack(storage)
            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(z-z0))), float(np.max(np.abs(result["a"]-a0))))
            z = z0
            if k:
                result = problem.evaluate(z, a0)
        elif k:
            result = problem.evaluate(z, result["a"])
        R = canonical_residual(z, result["gradient"])
        trace.append({"iteration": k, "fits": problem.fits, "objective": result["F"], "canonical_residual": R,
                      "relative_H1_error": result["relative_H1_error"], "z": z.tolist(),
                      "a": result["a"].tolist(), "modes": list(modes)})
        if R <= tolerance:
            status = "canonical_tolerance"
            break
        if problem.fits >= fit_budget:
            status = "fit_budget"
            break
        if k == max_steps:
            break
        alpha = 1.
        for backtracks in range(48):
            proposal = project(z, result["gradient"], alpha, coordinates)
            candidate = problem.evaluate(proposal, result["a"], need_gradient=False)
            distance2 = step_norm2(z, proposal, coordinates)
            required = result["F"]-.25/alpha*distance2
            if candidate["F"] <= required:
                break
            alpha *= .5
        else:
            status = "descent_unresolved"
            break
        worst_descent_violation = max(worst_descent_violation, candidate["F"]-required)
        trace[-1].update({"accepted_step": alpha, "backtracks": backtracks, "step_norm": np.sqrt(distance2),
                          "descent_margin": required-candidate["F"]})
        z, result = proposal, candidate
        if coordinates == "t":
            new_modes = ["ordinary" if mode == "stable" and t >= TAU_OUT else
                         "stable" if mode == "ordinary" and t <= TAU_IN else mode
                         for mode, t in zip(modes, z[:, 2])]
            for j in range(M):
                if new_modes[j] != modes[j]:
                    events.append({"iteration": k+1, "pair": j, "from": modes[j], "to": new_modes[j], "t": float(z[j, 2])})
            modes = new_modes
            storage = storage_pack(z, result["a"], modes)
    elapsed = time.perf_counter()-start
    final = trace[-1]
    return {"seed": seed, "method": "reversible_t" if coordinates == "t" else "stable_h",
            "status": status, "iterations": final["iteration"], "fits": problem.fits,
            "seconds": elapsed, "switches": len(events), "canonical_residual": final["canonical_residual"],
            "relative_H1_error": final["relative_H1_error"], "objective": final["objective"],
            "max_coefficient_residual": problem.max_fit_residual,
            "max_scaled_coefficient_residual": problem.max_fit_scaled_residual,
            "max_coordinate_roundtrip_error": max_roundtrip,
            "worst_descent_violation": worst_descent_violation, "events": events, "trace": trace}


def validate_formulas():
    z = tensor(initial_geometry(4101))
    matrix = gram(z, z).numpy()
    centers, transform = [], np.zeros((4, 4))
    for j in range(2):
        c, t, e = z[j, :2].numpy(), float(z[j, 2]), z[j, 3:5].numpy()
        h = np.sqrt(t)
        centers.extend([c+h*e, c-h*e])
        transform[2*j:2*j+2, 2*j:2*j+2] = [[.5, 1/(2*h)], [.5, -1/(2*h)]]
    delta = np.array(centers)[:, None, :]-np.array(centers)[None, :, :]
    rr = np.sum(delta*delta, axis=-1)
    raw = np.pi*np.exp(-rr/4)*(2-rr/4)
    mismatch = np.max(np.abs(matrix-transform.T@raw@transform))
    assert mismatch < 2e-13, mismatch
    p = Problem()
    start = initial_geometry(4101)
    ans = p.evaluate(start)
    fd_errors = []
    for j, col in [(0, 0), (0, 2), (1, 4)]:
        h = 2e-5
        zp, zm = start.copy(), start.copy()
        zp[j, col] += h
        zm[j, col] -= h
        fd = (p.evaluate(zp, ans["a"], False)["F"]-p.evaluate(zm, ans["a"], False)["F"])/(2*h)
        fd_errors.append(abs(fd-ans["gradient"][j, col]))
    assert max(fd_errors) < 2e-8, fd_errors
    target = p.evaluate(TARGET_Z, TARGET_A)
    assert target["F"] < 2e-15 and np.linalg.norm(target["gradient"]) < 2e-13
    return {"raw_analytic_gram_max_abs_difference": float(mismatch),
            "central_difference_gradient_errors": fd_errors,
            "target_objective": target["F"], "target_gradient_norm": float(np.linalg.norm(target["gradient"])),
            "quadrature_target_max_abs": float(p.target.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--fit-budget", type=int, default=6000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4101, 4102, 4103])
    parser.add_argument("--methods", nargs="+", choices=["t", "h"], default=["t", "h"])
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    checks = validate_formulas()
    results = []
    output = Path(__file__).parent/args.output
    settings = vars(args) | {"dimension": 2, "pairs": 2, "sigma": SIGMA, "lambda": LAMBDA,
              "target_z": TARGET_Z.tolist(), "target_coefficients": TARGET_A.tolist(),
              "midpoint_bounds": [LOW[:, :2].tolist(), HIGH[:, :2].tolist()], "t_bounds": [0., .36],
              "tau_in": TAU_IN, "tau_out": TAU_OUT, "coefficient_scaled_tolerance": COEFFICIENT_TOL,
              "alpha0": 1., "backtracking_factor": .5, "eta": .5, "quadrature_box": [-7., 7.],
              "quadrature": "positive tensor Gauss-Legendre, nonlinear term only",
              "H1_term": "exact analytic whole-space Gaussian correlations",
              "objective": "target-normalized PB Bregman energy with fixed discrete nonlinear forcing",
              "precision": "IEEE float64; no exact-arithmetic claim", "torch_threads": 1,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform()},
              "timing_note": "single-process serial timings; concurrent workstation activity is uncontrolled"}
    for seed in args.seeds:
        for method in args.methods:
            result = solve(seed, method, args.order, args.max_steps, args.fit_budget, args.tolerance)
            results.append(result)
            output.write_text(json.dumps({"settings": settings, "formula_checks": checks, "runs": results}, indent=2)+"\n")
            print(seed, result["method"], result["status"], "steps", result["iterations"], "fits", result["fits"],
                  "R", result["canonical_residual"], "error", result["relative_H1_error"], "seconds", result["seconds"], flush=True)
    fields = ["seed", "method", "status", "iterations", "fits", "seconds", "switches", "canonical_residual", "relative_H1_error",
              "max_scaled_coefficient_residual", "max_coordinate_roundtrip_error", "worst_descent_violation"]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
