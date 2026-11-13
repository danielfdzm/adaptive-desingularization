#!/usr/bin/env python3
"""Exploratory independent-grid / scan-frequency variant, all outcomes retained.

The grid is identical across source families and depends on neither reference
solutions nor source centers. No held-out seeds are used.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import run_chain as rc
from validate_saved import reference

Base=rc.ForcingSolver
class GridVariant(Base):
    def __init__(self,name,seed,method,nodes=768,maxfits=1500):
        super().__init__(name,seed,method,nodes,maxfits)
        P=rc.CASES[name]['capacity'];rng=np.random.default_rng(seed)
        starts=np.linspace(-.75,.75,P)+rng.uniform(-.15,.15,P)
        self.initial_centers=starts.tolist()
        for b,c in zip(self.blocks,starts):b['c']=np.array([c])
        self.scan_every=5

rc.ForcingSolver=GridVariant
rc.reference=reference
out=rc.HERE/'pilot_grid_variant'
out.mkdir(parents=True,exist_ok=True)
(out/'variant_protocol.json').write_text(json.dumps(dict(
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changes=['initial grid [-.75,.75] independent of all source centers','scan every5 iterations','staged BVP initialization only'],
    cases=['multipole4_nonlinear','multipole5_nonlinear'],
    seeds=[200,201,202],holdouts_used=False,
    interpretation='exploratory policy variant after broad-grid pilot failures, all retained'),indent=2))
sys.argv=[sys.argv[0],'--cases','multipole4_nonlinear','multipole5_nonlinear','--output',str(out)]
rc.main()
