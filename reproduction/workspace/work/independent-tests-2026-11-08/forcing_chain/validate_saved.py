#!/usr/bin/env python3
"""Independent staged BVP validation of saved chain endpoints; no optimization."""
import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
import run_chain as rc

@lru_cache(None)
def reference(name,L=12.,tol=1e-10):
    lam=rc.CASES[name]['reaction'];x=np.linspace(-L,L,1201)
    def ode(x,y):return np.vstack((y[1],y[0]+lam*(np.sinh(y[0])-y[0])-rc.load(name,x)))
    def bc(a,b):return np.array([a[1]-a[0],b[1]+b[0]])
    coarse=solve_bvp(ode,bc,x,np.zeros((2,len(x))),tol=1e-7,max_nodes=120000)
    if not coarse.success:raise RuntimeError('coarse: '+coarse.message)
    fine=solve_bvp(ode,bc,coarse.x,coarse.y,tol=tol,max_nodes=120000)
    if not fine.success:raise RuntimeError('fine: '+fine.message)
    return fine

def validate(path):
    data=json.loads(path.read_text());name=data['problem']
    if 'final_blocks' not in data:return dict(source=str(path),status='no_endpoint',original=data)
    s=rc.ForcingSolver(name,data['seed'],data['method'])
    s.blocks=[{k:np.array(v) if k in ('c','e','s','p') else v for k,v in b.items()} for b in data['final_blocks']]
    s.coef=np.array(data['final_coefficients'])
    base=reference(name);fine=reference(name,14.,3e-11)
    rows=[]
    for n in (1024,1536,2048):
        x,w=rc.quadrature(n);u,ux=fine.sol(x);v,vx=s.fields(x)
        hnorm=float(w @ (u*u+ux*ux));err=math.sqrt(float(w @ ((v-u)**2+(vx-ux)**2))/hnorm)
        f=rc.load(name,x);lam=rc.CASES[name]['reaction']
        energy=float(w @ (.5*(v*v+vx*vx)+lam*rc.rem(v)-f*v))
        refenergy=float(w @ (.5*(u*u+ux*ux)+lam*rc.rem(u)-f*u))
        rows.append(dict(nodes=n,error=err,energy=energy,reference_energy=refenergy,
                         nonlinear_to_quadratic_ratio=float(lam*w@rc.rem(u))/(.5*hnorm),reference_peak=float(np.max(np.abs(u)))))
    diff=base.sol(x)-fine.sol(x)
    return dict(source=str(path),problem=name,seed=data['seed'],method=data['method'],
        status='validated',error=rows[-1]['error'],checks=rows,
        reference_relative_H1_difference=math.sqrt(float(w@np.sum(diff*diff,axis=0))/hnorm),
        base_mesh_nodes=len(base.x),fine_mesh_nodes=len(fine.x),fits=data['fits'],completions=data['completions'],
        joins=data['joins'],original_stop=data['stop'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);args=ap.parse_args()
    out=args.input/'staged_validation';out.mkdir(exist_ok=True)
    for path in sorted((args.input/'runs').glob('*.json')):
        dst=out/path.name
        if dst.exists():continue
        try:r=validate(path)
        except Exception as exc:r=dict(source=str(path),status='validation_exception',exception=repr(exc))
        dst.write_text(json.dumps(rc.clean(r),indent=2))
        print(path.name,r.get('status'),r.get('error'),flush=True)

if __name__=='__main__':main()
