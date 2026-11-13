#!/usr/bin/env python3
"""Independent, forcing-defined 1D pilot; retain every result, no holdout use.

The existing pair-only solver is imported unchanged. Only its problem factory
and reference cache are replaced. Training uses forcing values, fixed spatial
quadrature, and perturbed grids; no reference solution enters optimization.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'adaptive-desingularization/final-experiments/pb-independent-1d'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('independent_multi', SOURCE / 'multi.py')
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)

# Tuples are (shape, center, width/radius, amplitude). Gaussian odd sources
# use z exp(-z²/2); compact sources use z(1-z²)^4 on |z|<1.
# These are chosen forcing formulas, not obtained by applying a PDE to a target.
CASES = {
    'two_gaussian_linear': dict(capacity=4, reaction=0., sources=[('gaussian', -.85, .20, 24.), ('gaussian', .90, .35, -20.)]),
    'two_gaussian_nonlinear': dict(capacity=4, reaction=10., sources=[('gaussian', -.85, .20, 48.), ('gaussian', .90, .35, -40.)]),
    'two_compact_linear': dict(capacity=4, reaction=0., sources=[('compact', -.80, .55, 80.), ('compact', .95, .70, -65.)]),
    'two_compact_nonlinear': dict(capacity=4, reaction=10., sources=[('compact', -.80, .55, 160.), ('compact', .95, .70, -130.)]),
    'three_mixed_linear': dict(capacity=6, reaction=0., sources=[('compact', -1.15, .5, 65.), ('gaussian', .05, .27, -18.), ('compact', 1.20, .65, 55.)]),
    'three_mixed_nonlinear': dict(capacity=6, reaction=10., sources=[('compact', -1.15, .5, 130.), ('gaussian', .05, .27, -36.), ('compact', 1.20, .65, 110.)]),
}
ACTIVE = None
REFERENCES = {}
original_reference = multi.reference

def source_values(x, parts):
    ans = np.zeros_like(x)
    for shape, center, width, amplitude in parts:
        z = (x - center) / width
        shape_value = z * np.exp(-.5*z*z) if shape == 'gaussian' else z * np.maximum(1-z*z, 0.)**4
        ans += amplitude * shape_value
    return ans

def problem(name):
    case = CASES[name]
    return .55, case['reaction'], lambda x: source_values(x, case['sources']), None

def cached_reference(load, lam, L=12., tol=1e-10):
    key = (ACTIVE, lam, L, tol)
    if key not in REFERENCES:
        REFERENCES[key] = original_reference(load, lam, L=L, tol=tol)
    return REFERENCES[key]

multi.problem = problem
multi.reference = cached_reference

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value

def diagnostics(name):
    sigma, lam, load, _ = problem(name)
    sol = cached_reference(load, lam)
    xx, ww = multi.quadrature(1536)
    u, ux = sol.sol(xx)
    quadratic = float(.5 * ww @ (u*u+ux*ux))
    nonlinear = float(lam * ww @ multi.rem(u))
    return dict(reference_peak=float(np.max(np.abs(u))),
                reference_H1_squared=2*quadratic,
                nonlinear_to_quadratic_energy_ratio=nonlinear/quadratic,
                source_squared_norm=float(ww @ load(xx)**2),
                reference_mesh_nodes=int(len(sol.x)))

def summarize(out):
    records = [json.loads(p.read_text()) for p in (out/'runs').glob('*.json')]
    summary = []
    for name in CASES:
        for method in ['ordinary','stable','adaptive']:
            rr=[r for r in records if r['problem']==name and r['method']==method]
            if not rr: continue
            valid=[r for r in rr if isinstance(r.get('error'), (int,float))]
            summary.append(dict(problem=name, method=method, runs=len(rr),
                errors=[r.get('error') for r in rr],
                median_error=float(np.median([r['error'] for r in valid])) if valid else None,
                median_fits=float(np.median([r['fits'] for r in valid])) if valid else None,
                accepted_completions=sum(r.get('completed_count',0) for r in rr),
                exceptions=len(rr)-len(valid)))
    (out/'summary.json').write_text(json.dumps(clean(summary), indent=2))

def main():
    global ACTIVE
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',nargs='+',default=list(CASES))
    ap.add_argument('--seeds',nargs='+',type=int,default=[200,201,202])
    ap.add_argument('--nodes',type=int,default=768)
    ap.add_argument('--max-fits',type=int,default=800)
    ap.add_argument('--output',type=Path,default=HERE/'pilot')
    args=ap.parse_args()
    if any(900<=s<=909 for s in args.seeds):
        raise ValueError('Holdout seeds are reserved; freeze a separate confirmation protocol first.')
    out=args.output; (out/'runs').mkdir(parents=True,exist_ok=True)
    manifest=dict(arguments={**vars(args),'output':str(out)}, cases=CASES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'multi.py',SOURCE/'run.py']},
        python=platform.python_version(), role='exploratory screen, all outcomes retained',
        starts='linspace(-1.5,1.5,P) + independent uniform(-.15,.15) jitter',
        solver='fixed adjacent pairs; no regrouping/insertion/pruning',
        timing='sequential process, includes post-run validation; cached reference makes runtimes noncomparable')
    protocol=out/'protocol.json'
    if protocol.exists():
        old=json.loads(protocol.read_text())
        if old['script_sha256']!=manifest['script_sha256'] or old['source_hashes']!=manifest['source_hashes']:
            raise ValueError('Source changed: use a fresh output directory.')
    else: protocol.write_text(json.dumps(manifest,indent=2))
    for name in args.cases:
        ACTIVE=name
        case=CASES[name]
        for seed in args.seeds:
            for method in ['ordinary','stable','adaptive']:
                path=out/'runs'/f'{name}-{seed}-{method}.json'
                if path.exists(): continue
                t0=time.monotonic()
                try:
                    r=multi.run_one(name,case['capacity'],seed,method,args.nodes,args.max_fits)
                    r['reference_diagnostics']=diagnostics(name)
                except Exception as exc:
                    r=dict(problem=name,capacity=case['capacity'],seed=seed,method=method,status='exception',exception=repr(exc),seconds=time.monotonic()-t0)
                path.write_text(json.dumps(clean(r),indent=1))
                summarize(out)
                print(name,seed,method,'error',r.get('error'),'fits',r.get('fits'),'completed',r.get('completed_count'),'status',r['status'],flush=True)

if __name__=='__main__': main()
