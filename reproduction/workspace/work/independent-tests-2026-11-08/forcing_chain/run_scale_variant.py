#!/usr/bin/env python3
"""Exploratory forcing-width variant; PDE solution still independently solved.

Uses a prescribed fourth-order multipole source of width .55, equal to the
already-fixed model scale, with no exact solution or target-centered starts.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import run_chain as rc
from validate_saved import reference as staged_reference

for label,lam,amp in [('linear',0.,12.),('nonlinear',10.,60.)]:
    rc.CASES['multipole4_scale055_'+label]=dict(order=4,source_scale=.55,source_center=.23,amplitude=amp,reaction=lam,capacity=3)

Base=rc.ForcingSolver
class GridVariant(Base):
    def __init__(self,name,seed,method,nodes=768,maxfits=1500):
        super().__init__(name,seed,method,nodes,maxfits)
        P=rc.CASES[name]['capacity'];rng=np.random.default_rng(seed)
        starts=np.linspace(-.75,.75,P)+rng.uniform(-.15,.15,P)
        self.initial_centers=starts.tolist()
        for b,c in zip(self.blocks,starts):b['c']=np.array([c])
        self.scan_every=5

def resolved_reference(name,L=12.,tol=1e-10):
    return staged_reference(name,L,1e-9 if L<=12. else 3e-10)

rc.ForcingSolver=GridVariant
rc.reference=resolved_reference
out=rc.HERE/'pilot_scale_variant'
out.mkdir(parents=True,exist_ok=True)
(out/'variant_protocol.json').write_text(json.dumps(dict(
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changes=['forcing Gaussian width .55 matches previously fixed model scale','initial grid [-.75,.75] independent of source center',
             'scan every5 iterations','reference tol1e-9 on12 and3e-10 on14'],
    cases=['multipole4_scale055_linear','multipole4_scale055_nonlinear'],
    seeds=[200,201,202],holdouts_used=False,
    interpretation='exploratory width variant after high-error narrow-source pilots, all retained'),indent=2))
sys.argv=[sys.argv[0],'--cases','multipole4_scale055_linear','multipole4_scale055_nonlinear','--output',str(out)]
rc.main()
