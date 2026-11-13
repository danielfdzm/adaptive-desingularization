#!/usr/bin/env python3
"""Exploratory 1D manufactured recovery: iid starts independent of target."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import importlib.util
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from scipy.special import roots_legendre
_shared_spec = importlib.util.spec_from_file_location("blind_higher_order_shared", Path(__file__).with_name("run.py"))
shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(shared)

HERE, ROOT, EXP, PB = shared.HERE, shared.ROOT, shared.EXP, shared.PB
cs, ck, save, clean, bregman_parts = shared.cs, shared.ck, shared.save, shared.clean, shared.bregman_parts


def rule(n, box):
    x, w = roots_legendre(n)
    return box*x[:, None], box*w


def problem(order, seed, nodes=384, lam=10.0, max_fits=1200):
    sigma = .65
    target = (np.array([[.23]]), np.array([[1.0]]), np.array([order]), np.array([sigma**order]))
    starts = np.random.default_rng(seed).uniform(-1.2, 1.2, (order + 1, 1))
    X, W = rule(nodes, 10.)
    return dict(target=target, starts=starts, X=X, W=W, sigma=sigma, lam=lam, max_fits=max_fits)


def run_one(task):
    order, seed, method, nodes, lam, max_fits = task
    P = problem(order, seed, nodes, lam, max_fits)
    solver = cs.Solver(1, P["sigma"], P["target"], P["starts"], method, P["X"], P["W"],
                       lam=lam, max_fits=max_fits, box=3.)
    r = solver.run()
    r.update(case=f"blind1d_order{order}", order=order, seed=seed, lambda_value=lam, sigma=P["sigma"],
             target=[a.tolist() for a in P["target"]], initial_centers=P["starts"].tolist(),
             width=order + 1, manufactured_target=True, independent_initialization=True)
    target2 = float(ck.mp_norm2(solver.target_np, solver.sigma, 1, dps=80))
    quadratic_gap = .5*r["final_relative_error"]**2*target2
    C, U, O, WS = cs.primitives(solver.blocks, 1)
    weights = WS @ torch.as_tensor(solver.coef)
    validation = []
    for n, box in ((nodes, 10.), (2*nodes, 10.), (2*nodes, 12.)):
        X, W = rule(n, box)
        u = (ck.point_values(torch.as_tensor(X), *solver.target[:3], solver.sigma) @ solver.target[3]).numpy()
        v = (ck.point_values(torch.as_tensor(X), C, U, O, solver.sigma) @ weights).numpy()
        gap, _, _ = bregman_parts(v, u)
        Fu, _, _ = bregman_parts(u, np.zeros_like(u))
        nonlinear_gap = float(lam*(W@gap))
        validation.append(dict(nodes=n, box=box, quadratic_gap=quadratic_gap, nonlinear_gap=nonlinear_gap,
                               total_gap=quadratic_gap+nonlinear_gap,
                               target_nonlinear_over_quadratic=float(lam*(W@Fu)/(.5*target2))))
    blocks = [{k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in b.items()}
              for b in solver.blocks]
    M, b, Phi = (x.detach().numpy() for x in cs.assemble(blocks, 1, solver.sigma, solver.X, solver.target))
    _, residual, _ = bregman_parts(Phi@solver.coef, solver.Uq)
    grad = M@solver.coef-b+Phi.T@(solver.Wq*residual)
    r["validation"] = dict(quadrature=validation,
            normalized_coefficient_kkt=float(np.max(np.abs(grad)/np.sqrt(np.diag(M)))/math.sqrt(target2)),
            maximum_relative_gap_change=max(abs(x["total_gap"]/validation[-1]["total_gap"]-1)
                for x in validation[:-1]) if validation[-1]["total_gap"]>0 else 0.)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(200, 206)))
    ap.add_argument("--methods", nargs="+", default=["standard_restart", "stable", "adaptive"])
    ap.add_argument("--nodes", type=int, default=384)
    ap.add_argument("--lam", type=float, default=10.)
    ap.add_argument("--max-fits", type=int, default=1200)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--output", type=Path, default=HERE/"pilot1d")
    args=ap.parse_args()
    if any(s>=900 for s in args.seeds):
        ap.error("Holdout seeds intentionally disabled for exploration.")
    args.output.mkdir(parents=True, exist_ok=True)
    sources=[Path(__file__),HERE/"run.py",EXP/"chain_solver.py",EXP/"chain_kernels.py",PB/"fit.py",PB/"quadrature.py"]
    protocol=dict(purpose="manufactured 1D recovery from iid U[-1.2,1.2] centers",
                  parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  target_shift=.23,sigma=.65,target_amplitudes="sigma**order",source_behavior="unmodified chain_solver",
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                  stage="exploratory extension after recording all 2D pilot outcomes")
    pp=args.output/"protocol.json"
    if pp.exists() and json.loads(pp.read_text())!=clean(protocol):
        raise RuntimeError("Protocol mismatch: use new output directory")
    save(pp,protocol)
    tasks=[(r,s,m,args.nodes,args.lam,args.max_fits) for r in args.orders for s in args.seeds for m in args.methods]
    name=lambda t:f"order{t[0]}_seed{t[1]}_{t[2]}"
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,t):t for t in tasks if not(args.output/f"{name(t)}.json").exists()}
        for future in as_completed(futures):
            t=futures[future]
            try:r=future.result()
            except Exception as exc:r=dict(case=f"blind1d_order{t[0]}",order=t[0],seed=t[1],method=t[2],stop="exception",exception=repr(exc))
            save(args.output/f"{name(t)}.json",r)
            print(name(t),"error",r.get("final_relative_error"),"fits",r.get("fits"),"completed",r.get("completions"),"stop",r["stop"],flush=True)

if __name__=="__main__":main()
