"""Matched geometric-coordinate control: optimize log(h) instead of h.

Only positive finite half-gaps are reparameterized; physical states, bounds,
coefficient fits, completion gates and work budgets remain unchanged.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import multi


def logarithmic_minimize(fun,x0,*,jac,method,bounds,options):
    mask=np.array([lo>0 for lo,hi in bounds])
    y=np.asarray(x0).copy();y[mask]=np.log(y[mask])
    bb=[(np.log(lo),np.log(hi)) if selected else (lo,hi)
        for selected,(lo,hi) in zip(mask,bounds)]
    def unpack(yy):
        x=np.asarray(yy).copy();x[mask]=np.exp(x[mask]);return x
    def wrapped(yy):
        x=unpack(yy);value,gradient=fun(x)
        gradient=gradient.copy();gradient[mask]*=x[mask]
        return value,gradient
    result=scipy_minimize(wrapped,y,jac=jac,method=method,bounds=bb,options=options)
    result.x=unpack(result.x)
    result.jac[mask]/=result.x[mask]
    return result


if __name__=='__main__':
    argv=sys.argv[1:]
    output=argv[argv.index('--output')+1] if '--output' in argv else 'multi-results'
    out=multi.HERE/output;out.mkdir(parents=True,exist_ok=True)
    (out/'coordinate-protocol.json').write_text(json.dumps(dict(gap_coordinate='log(h)',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_bounds=[1e-8,1.5],ordinary='unchanged ordinary center coordinates',selection='All problems, capacities, fixed seeds and methods are retained.'),indent=2))
    multi.minimize=logarithmic_minimize
    multi.main()
