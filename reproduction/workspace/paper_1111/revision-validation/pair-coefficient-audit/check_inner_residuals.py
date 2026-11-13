"""Re-evaluate normalized coefficient gradients at the 30 saved PG-t endpoints.

Run from any directory; no coefficient fits or optimization are performed.
This is a float64 endpoint diagnostic, not a validated rounding-error bound.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "work/alg41-pb-cohort-2026-11-12/alg41.py").is_file())
SOURCE = ROOT / "work/alg41-pb-cohort-2026-11-12/alg41.py"
spec = importlib.util.spec_from_file_location("alg41_inner_audit", SOURCE)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)

rows = []
for path in sorted((SOURCE.parent / "results/pg_t").glob("*.json")):
    r = json.loads(path.read_text())
    s = r["settings"]
    problem = a.Problem(r["d"], r["geometry"], r["seed"],
                        power=s["quadrature_power"],
                        quadrature_seed=s["quadrature_seed"],
                        reaction=s["reaction"])
    z = {k: a.torch.as_tensor(r["final_state"][k]) for k in ["beta", "t", "e", "c"]}
    coef = a.np.asarray(r["final_state"]["coef"])
    blocks = problem.blocks_from(z["beta"], z["t"], z["e"], z["c"])
    M = a.sg.block_gram(blocks).numpy()
    M = 0.5 * (M + M.T)
    b = (a.sg.block_gram(blocks, problem.tb) @ a.torch.as_tensor(problem.tc)).numpy()
    Phi = a.sg.column_values(blocks, problem.X).numpy()
    _, residual, _ = a.bregman_parts(Phi @ coef, problem.U)
    gradient = (M @ coef - b + Phi.T @ (problem.W * residual)) / problem.target2
    rows.append(dict(record=str(path.relative_to(ROOT)),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     dimension=r["d"],
                     coefficient_gradient_norm2=float(a.np.linalg.norm(gradient)),
                     coefficient_gradient_max=float(a.np.max(a.np.abs(gradient))),
                     outer_residual=r["final_residual"]))
assert len(rows) == 30
result = dict(scope="Float64 endpoint coefficient gradients, energy divided by target H1 norm squared; no refits.",
              source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              rows=rows,
              max_norm2=max(r["coefficient_gradient_norm2"] for r in rows),
              max_max=max(r["coefficient_gradient_max"] for r in rows))
print(json.dumps(result, indent=2))
