#!/usr/bin/env python3
"""Target/isometry, high-dimensional Gaussian calculus, and envelope checks."""
from __future__ import annotations
import json
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from run import Solver,Geometry
from config import embedding,configuration,target_blocks
from kernels import block_gram,validation_checks,_mp_primitives,_mp_inner

def main():
    rng=np.random.default_rng(7648)
    checks=[]
    for d in (5,10,20):
        for geometry in (0,1):
            q=embedding(d,geometry)
            orth=float(np.linalg.norm(q.T@q-np.eye(5)))
            assert orth<1e-12
            assert configuration(d,geometry)['geometry_rank']==5
            blocks,coef=target_blocks(d,geometry)
            for b in blocks[:4]:b.update(completed=False,h=.27)
            a=block_gram(blocks).numpy();b=block_gram(blocks,order=16).numpy()
            gram=float(np.linalg.norm(a-b)/np.linalg.norm(b))
            with mp.workdps(80):
                primitives=_mp_primitives(blocks,coef)
                exact=float(_mp_inner(primitives,primitives,.45))
            discrepancy=abs(float(coef@a@coef)-exact)/exact
            assert gram<1e-12 and discrepancy<1e-12
            checks.append(dict(dimension=d,geometry_id=geometry,orthogonality_error=orth,
                relative_gram_order_error=gram,relative_high_precision_quadratic_error=discrepancy))
    s=Solver(10,901,'adaptive',8,200,0,8128,100.)
    s.fit_current();s.regroup();s.fit_current()
    geo=Geometry(s.blocks,s.d)
    _,grad=s.evaluate(geo.x,geo)
    direction=rng.normal(size=len(geo.x));direction/=np.linalg.norm(direction)
    h=1e-5
    fd=(s.evaluate(geo.x+h*direction,geo)[0]-s.evaluate(geo.x-h*direction,geo)[0])/(2*h)
    analytic=float(grad@direction)
    relative=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-8)
    assert relative<2e-4,(fd,analytic,relative)
    out=dict(passed=True,original_kernel_checks=validation_checks(),high_dimensional_checks=checks,
             envelope_gradient=dict(dimension=10,reaction_strength=100.,power=8,finite_difference=fd,
                                    analytic=analytic,relative_discrepancy=relative))
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
