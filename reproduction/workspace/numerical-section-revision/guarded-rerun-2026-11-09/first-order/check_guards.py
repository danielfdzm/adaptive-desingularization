#!/usr/bin/env python3
"""Targeted checks of state retention at the pair-direction guard."""
import json
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source/final-experiments/pb'))
import numpy as np
import torch
import run
from kernels import block_gram, column_values

def bare(distance):
    s=run.Solver.__new__(run.Solver)
    s.blocks=[dict(kind='single',id='0',c=np.array([0.,0.])),
              dict(kind='single',id='1',c=np.array([distance,0.]))]
    s.coef=np.array([1.,-1.]);s.rho0=run.kappa(2,0);s.rho1=2*run.kappa(2,1)
    s.events=[];s.fits=0;s.cache=None;s.warm=None
    return s

def main():
    threshold=32*(np.finfo(float).eps/2)*run.SIGMA
    checks=[]
    for distance in (0.,threshold/2,threshold,float('nan'),float('inf')):
        s=bare(distance)
        assert not s.regroup()
        assert len(s.blocks)==2 and all(b['kind']=='single' for b in s.blocks)
        assert np.array_equal(s.coef,np.array([1.,-1.]))
        checks.append(dict(distance=distance if np.isfinite(distance) else str(distance),
                           retained=True,guard_counts=s.guard_counts))
    s=bare(2*threshold)
    assert s.regroup() and s.blocks[0]['kind']=='pair'
    assert np.all(np.isfinite(s.blocks[0]['e']))
    checks.append(dict(distance=2*threshold,paired=True,guard_counts=s.guard_counts))
    h=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    finite=[dict(kind='pair',c=np.array([.1,-.1]),e=np.array([1.,0.]),h=h,completed=False)]
    completed=[dict(finite[0],h=0.,completed=True)]
    X=torch.tensor([[.2,.3],[-.1,.4]],dtype=torch.float64)
    gram_difference=float(torch.max(torch.abs(block_gram(finite)-block_gram(completed))).detach())
    value_difference=float(torch.max(torch.abs(column_values(finite,X)-column_values(completed,X))).detach())
    assert gram_difference<1e-12 and value_difference<1e-14
    derivative=torch.autograd.grad(block_gram(finite).sum()+column_values(finite,X).sum(),h)[0]
    assert torch.isfinite(derivative)
    result=dict(pair_checks=checks,h_zero_extension=dict(gram_difference=gram_difference,
        value_difference=value_difference,finite_autodiff=True),
        directional_joins='not implemented in these pair-only solvers; C.4 is inapplicable')
    (HERE/'guard-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
