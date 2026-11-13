#!/usr/bin/env python3
"""Holdout endpoint sensitivity to doubled internal divided-difference quadrature."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import roots_legendre

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("blind_basis_shared",HERE/"run.py")
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
cs,ck=shared.cs,shared.ck
x,w=roots_legendre(768);X=torch.as_tensor(12*x[:,None]);W=12*w

def sampled(blocks,coeff,nq):
    cs.NQ=nq
    cs.GL_X,cs.GL_W=roots_legendre(2*nq)
    C,U,O,S=cs.primitives(blocks,1)
    ww=S@torch.as_tensor(coeff)
    return ((ck.point_values(X,C,U,O,.65)@ww).numpy(),
            ((ck.point_values(X,C,U,O+1,.65)*U[:,0][None,:])@ww).numpy())

rows=[]
for path in sorted((HERE/"holdout").glob("order*_seed*.json")):
    r=json.loads(path.read_text())
    if "final_blocks" not in r:continue
    blocks=[{k:np.asarray(v,float) if isinstance(v,list) else v for k,v in b.items()} for b in r["final_blocks"]]
    u,ux=sampled(blocks,r["final_coefficients"],4)
    v,vx=sampled(blocks,r["final_coefficients"],8)
    tc,tu,to,tw=(np.asarray(a) for a in r["target"])
    target2=float(torch.as_tensor(tw)@ck.h1_gram(torch.as_tensor(tc),torch.as_tensor(tu),to,
                    torch.as_tensor(tc),torch.as_tensor(tu),to,.65)@torch.as_tensor(tw))
    diff=float(np.sqrt(W@((u-v)**2+(ux-vx)**2)/target2))
    rows.append(dict(order=r["order"],seed=r["seed"],method=r["method"],basis_change_relative_H1=diff,
                     reported_error=r["final_relative_error"]))
report=dict(rows=rows,max_basis_change_relative_H1=max(r["basis_change_relative_H1"] for r in rows),
            description="Fixed stored geometry and coefficients; pair quadrature 8 to16 and joined-block quadrature4 to8, evaluated with768-point GL on[-12,12]. Double-precision sensitivity check, not interval certificate.")
shared.save(HERE/"basis_validation.json",report)
print(report["max_basis_change_relative_H1"])
