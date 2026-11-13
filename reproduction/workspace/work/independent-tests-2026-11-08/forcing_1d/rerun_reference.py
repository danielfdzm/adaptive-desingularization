#!/usr/bin/env python3
"""Rerun the three-source nonlinear case with a larger reference mesh cap.

The original pilot's nine reference failures remain in pilot/runs. The solver
and mathematical reference problem/tolerances are unchanged; only the BVP cap
increases from 30,000 to 120,000 nodes.
"""
import hashlib
import json
import sys
from pathlib import Path
import run_screen as screen
import numpy as np
from scipy.integrate import solve_bvp

def larger_reference(load, lam, L=12., tol=1e-10):
    x=np.linspace(-L,L,601)
    def fun(x,y):
        return np.vstack((y[1],y[0]+lam*(np.sinh(y[0])-y[0])-load(x)))
    def bc(a,b):
        return np.array([a[1]-a[0],b[1]+b[0]])
    sol=solve_bvp(fun,bc,x,np.zeros((2,len(x))),tol=tol,max_nodes=120000)
    if not sol.success: raise RuntimeError(sol.message)
    return sol

screen.original_reference=larger_reference
out=screen.HERE/'pilot_reference_repair'
out.mkdir(parents=True,exist_ok=True)
(out/'reference_change.json').write_text(json.dumps(dict(
    change='Only BVP reference max_nodes 30000 -> 120000',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    retained_original='pilot/runs/three_mixed_nonlinear-*'),indent=2))
sys.argv=[sys.argv[0],'--cases','three_mixed_nonlinear','--output',str(out)]
screen.main()
