#!/usr/bin/env python3
"""Independent quadrature and final-chart audit; excluded from solve timings."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np
import torch
from scipy.special import roots_legendre
import chain_experiments as ce
import chain_solver as cs
import chain_kernels as ck
from quadrature import make_quadrature, bregman_parts


def load_blocks(row):
    return [{k: np.asarray(v, float) if isinstance(v, list) else v for k, v in b.items()}
            for b in row["final_blocks"]]


def values(blocks, coef, X, d, sigma):
    C, U, O, WS = cs.primitives(blocks, d, sigma)
    weights = WS @ torch.as_tensor(coef)
    return np.concatenate([(ck.point_values(torch.as_tensor(X[j:j + 2048]), C, U, O, sigma)
                            @ weights).numpy() for j in range(0, len(X), 2048)])


def values_and_gradients(blocks, coef, X, d, sigma):
    C, U, O, WS = cs.primitives(blocks, d, sigma)
    weights = WS @ torch.as_tensor(coef)
    result, gradients = [], []
    for j in range(0, len(X), 512):
        points = torch.as_tensor(X[j:j + 512])
        primitive_values = ck.point_values(points, C, U, O, sigma)
        lower_values = ck.point_values(points, C, U, np.maximum(O - 1, 0), sigma)
        result.append((primitive_values @ weights).numpy())
        components = []
        for coordinate in range(d):
            derivative = -(points[:, coordinate, None] - C[None, :, coordinate]) / sigma ** 2 * primitive_values
            derivative -= torch.as_tensor(O)[None, :] * U[None, :, coordinate] / sigma ** 2 * lower_values
            components.append((derivative @ weights).numpy())
        gradients.append(np.stack(components, axis=1))
    return np.concatenate(result), np.concatenate(gradients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-historical-standard", action="store_true")
    args = parser.parse_args()
    inputs = sorted((HERE / "accuracy-higher").glob("higher_*.json"))
    if args.include_historical_standard:
        original = HERE.parents[2] / "revision-2026-11/experiments/results/chain-higher-d5"
        inputs += sorted(original.glob("higher_*_standard.json"))
    output = HERE / "endpoint-audit.json"
    done = json.loads(output.read_text()) if output.exists() else {"rows": {}, "summary": {}}
    done["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    done["solver_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / "src").glob("*.py"))}
    cache = {}
    for path in inputs:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name in done["rows"] and done["rows"][path.name]["input_sha256"] == digest:
            continue
        row = json.loads(path.read_text())
        d, geometry, seed = row["d"], row["geometry"], row["seed"]
        problem = ce.problem_higher(d, geometry, seed)
        sigma = problem["sigma"]
        target = tuple(torch.as_tensor(a) if i != 2 else a for i, a in enumerate(problem["target"]))
        if geometry not in cache:
            centers = np.unique(np.round(problem["target"][0], 12), axis=0)
            rules = {"training": (problem["X"], problem["W"]),
                     "independent": make_quadrature(centers, sigma=sigma, power=14, seed=91000 + geometry)[:2],
                     "refined": make_quadrature(centers, sigma=sigma, power=16, seed=91000 + geometry)[:2],
                     "narrow": make_quadrature(centers, sigma=sigma / math.sqrt(2), power=14, seed=92000 + geometry)[:2]}
            target_values = {name: (ck.point_values(torch.as_tensor(X), *target[:3], sigma) @ target[3]).numpy()
                             for name, (X, W) in rules.items()}
            norm2 = float(target[3] @ ck.h1_gram(*target[:3], *target[:3], sigma) @ target[3])
            cache[geometry] = rules, target_values, norm2
        rules, target_values, norm2 = cache[geometry]
        blocks, coefficients = load_blocks(row), np.asarray(row["final_coefficients"])
        cs.validate_blocks(blocks, sigma)
        quadratic = .5 * row["final_relative_error"] ** 2 * norm2
        nonlinear = {}
        for name, (X, W) in rules.items():
            V = values(blocks, coefficients, X, d, sigma)
            nonlinear[name] = float(W @ bregman_parts(V, target_values[name])[0])
        M, rhs, Phi = (a.numpy() for a in cs.assemble(blocks, d, sigma, torch.as_tensor(problem["X"]), target))
        residual = bregman_parts(Phi @ coefficients, target_values["training"])[1]
        gradient = M @ coefficients - rhs + Phi.T @ (problem["W"] * residual)
        stationarity = float(np.linalg.norm(gradient, ord=np.inf) / max(norm2, 1e-300))
        independent_X, independent_W = rules["independent"]
        sample_stride = max(1, len(independent_X) // 1024)
        sample, sample_weights = independent_X[::sample_stride], independent_W[::sample_stride]
        ordinary_values, ordinary_gradients = values_and_gradients(blocks, coefficients, sample, d, sigma)
        old_nq, old_x, old_w = cs.NQ, cs.GL_X, cs.GL_W
        try:
            cs.NQ = 8
            cs.GL_X, cs.GL_W = roots_legendre(16)
            refined_values, refined_gradients = values_and_gradients(blocks, coefficients, sample, d, sigma)
        finally:
            cs.NQ, cs.GL_X, cs.GL_W = old_nq, old_x, old_w
        basis_change = float(np.max(np.abs(refined_values - ordinary_values)))
        basis_relative = basis_change / max(float(np.max(np.abs(ordinary_values))), 1e-300)
        sample_h1_change2 = sample_weights @ ((refined_values - ordinary_values) ** 2
            + np.sum((refined_gradients - ordinary_gradients) ** 2, axis=1))
        sample_h1_norm2 = sample_weights @ (ordinary_values ** 2 + np.sum(ordinary_gradients ** 2, axis=1))
        sampled_h1_relative = float(np.sqrt(sample_h1_change2 / max(sample_h1_norm2, 1e-300)))
        record = {"input_sha256": digest, "method": row["method"], "geometry": geometry, "seed": seed,
                  "all_final_charts_admissible": True, "quadratic": quadratic, "nonlinear": nonlinear,
                  "total": {key: quadratic + value for key, value in nonlinear.items()},
                  "coefficient_gradient_inf_normalized": stationarity,
                  "basis_refinement_sample_count": len(sample),
                  "basis_refinement_max_value_change": basis_change,
                  "basis_refinement_relative_max_value_change": basis_relative,
                  "basis_refinement_sampled_H1_relative": sampled_h1_relative,
                  "basis_refinement_note": "1024 fixed independent Sobol-mixture nodes and positive weights; sampled value-and-gradient diagnostic, not a rigorous whole-space H1 bound",
                  "note": "mp80 H1 endpoint error is stored in run; nonlinear terms use independent numerical quadrature, not certified integration bounds."}
        done["rows"][path.name] = record
        output.write_text(json.dumps(done, indent=2) + "\n")
        print(path.name, "stationarity", stationarity, "gap", record["total"], flush=True)
    rows = list(done["rows"].values())
    same_order = 0
    starts = sorted({(r["geometry"], r["seed"]) for r in rows})
    for start in starts:
        group = [r for r in rows if (r["geometry"], r["seed"]) == start]
        order = {tuple(r["method"] for r in sorted(group, key=lambda x: x["total"][rule]))
                 for rule in ("training", "independent", "refined", "narrow")}
        same_order += len(order) == 1
    done["complete"] = len(rows) == 30
    done["expected_rows"] = 30
    done["summary"] = {"runs": len(rows), "all_final_charts_admissible": all(r["all_final_charts_admissible"] for r in rows),
                       "starts": len(starts), "starts_with_unchanged_method_order": same_order,
                       "max_coefficient_gradient_inf_normalized": max(r["coefficient_gradient_inf_normalized"] for r in rows),
                       "max_basis_refinement_relative_value_change": max(r["basis_refinement_relative_max_value_change"] for r in rows),
                       "max_basis_refinement_sampled_H1_relative": max(r["basis_refinement_sampled_H1_relative"] for r in rows),
                       "max_relative_total_gap_change": max(abs(r["total"][rule] - r["total"]["training"]) / max(abs(r["total"]["training"]), 1e-300)
                           for r in rows for rule in ("independent", "refined", "narrow"))}
    output.write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
