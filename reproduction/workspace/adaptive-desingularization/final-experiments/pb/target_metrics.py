#!/usr/bin/env python3
"""Dimension-invariant peak estimates for the underlying five-dimensional targets.

The peak estimate is a multistart optimization lower bound, not a certificate.
An elementary sum-of-term sup bound is also supplied. Isometric embedding
multiplies the 5D target by exp(-|x_perp|²/(2σ²)), so its exact supremum is
unchanged with ambient dimension, even though sampled maxima usually decrease.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from config import base_geometry,COEFFICIENTS,SIGMA

def main():
    rows=[]
    for geometry in (0,1):
        centers,directions=base_geometry(geometry)
        a=np.r_[COEFFICIENTS[:8:2],COEFFICIENTS[8:]]
        b=COEFFICIENTS[1:8:2]
        def value_gradient(x):
            z=x[None,:]-centers
            gauss=np.exp(-np.sum(z*z,axis=1)/(2*SIGMA**2))
            amp=a.copy();amp[:4]+=b*np.einsum('ij,ij->i',z[:4],directions)/SIGMA**2
            grad=-np.sum((gauss*amp)[:,None]*z,axis=0)/SIGMA**2
            grad+=np.sum((gauss[:4]*b)[:,None]*directions,axis=0)/SIGMA**2
            return float(gauss@amp),grad
        rng=np.random.default_rng(33102+geometry)
        starts=np.vstack((centers,centers[:4]+SIGMA*directions,centers[:4]-SIGMA*directions,rng.normal(size=(32,5))*.5))
        solutions=[]
        for sign in (-1.,1.):
            for x in starts:
                sol=minimize(lambda y:(-sign*value_gradient(y)[0],-sign*value_gradient(y)[1]),x,
                             jac=True,method='L-BFGS-B',options={'gtol':1e-11,'ftol':1e-15,'maxiter':1000})
                solutions.append((abs(value_gradient(sol.x)[0]),sol.x,sol.success))
        peak,x,success=max(solutions,key=lambda z:z[0])
        upper=float(np.sum(abs(a))+np.sum(abs(b))*math.exp(-.5)/SIGMA)
        rows.append(dict(geometry_id=geometry,estimated_peak_lower_bound=peak,
                         peak_location_in_base_coordinates=x.tolist(),candidate_optimizer_success=bool(success),
                         starts_per_sign=len(starts),elementary_sup_upper_bound=upper,
                         exact_sup_invariant_under_ambient_embedding=True))
    out=dict(method='deterministic multistart optimization; not a global certificate',targets=rows)
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
