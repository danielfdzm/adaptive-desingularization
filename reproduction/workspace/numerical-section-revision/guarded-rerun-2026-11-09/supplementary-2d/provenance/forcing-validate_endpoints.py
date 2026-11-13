"""Evaluate pilot endpoints directly on a refined Fourier reference grid.

This avoids the coarse cubic reference interpolation used for online diagnostics.
Original records remain unchanged. Compact-source training needs further quadrature
refinement, so its outcomes are explicitly marked as underresolved exploratory data.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import resample
import run as ex

HERE=Path(__file__).resolve().parent
out={'cases':{},'endpoints':[]}
for case in ['dipole_nonlinear','two_nonlinear','compact_nonlinear',
             'dipole_medium','two_medium','wave_nonlinear','multi6_nonlinear','multi6_redundant']:
    files=[p for d in ['pilot-smoke','pilot-smooth','pilot-multi6'] for p in (HERE/d).glob(f'{case}_*.json')]
    if not files: continue
    cfg=ex.CASES[case]
    x,u,ux,uy,meta=ex.reference_cached(case,cfg,n=512,L=10.)
    X,Y=np.meshgrid(x,x,indexing='ij'); pts=np.column_stack([X.ravel(),Y.ravel()])
    dx=x[1]-x[0]; norm2=meta['norm2']
    # A resampled interpolation comparison isolates the large online cubic error.
    xc,uc,uxc,uyc,mc=ex.reference_cached(case,cfg)
    VX,VW=ex.ind.rule(192,6.)
    refs=[]
    for axis,fields,L in [(xc,[uc,uxc,uyc],8.),(x,[u,ux,uy],10.)]:
        nup=1024 if L==8 else 1280
        xx=np.linspace(-L,L,nup,endpoint=False)
        vals=[]
        for z in fields:
            z=resample(resample(z,nup,axis=0),nup,axis=1)
            vals.append(RectBivariateSpline(xx,xx,z).ev(VX[:,0],VX[:,1]))
        refs.append(np.array(vals))
    rel=np.sqrt((VW@np.sum((refs[0]-refs[1])**2,axis=0))/(VW@np.sum(refs[1]*refs[1],axis=0)))
    out['cases'][case]={'reference':meta,'coarse_fine_H1_after_spectral_upsampling':float(rel),
                        'training_adequacy':'underresolved compact-source pilot' if cfg['shape']=='compact' else 'smooth source; see energy checks'}
    for path in files:
        r=json.loads(path.read_text())
        if 'blocks' not in r: continue
        vv,gg=ex.values_grad(r['blocks'],np.array(r['coefficients']),pts)
        err2=float(np.sum((vv-u.ravel())**2+(gg[:,0]-ux.ravel())**2+(gg[:,1]-uy.ravel())**2)*dx*dx)
        out['endpoints'].append({'file':str(path.relative_to(HERE)),
                                'relative_H1_direct_refined':float(np.sqrt(err2/norm2)),
                                'online_relative_H1':r['final_relative_error']})
    print(case,'reference difference',rel,'endpoints',len(files),flush=True)
    HERE.joinpath('endpoint_validation.json').write_text(json.dumps(out,indent=2))
