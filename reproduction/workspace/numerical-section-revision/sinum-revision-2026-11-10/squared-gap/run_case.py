#!/usr/bin/env python3
"""Run T with the frozen NC protocol and an exact squared-gap chart."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dimension',type=int,required=True)
    ap.add_argument('--geometry',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timing-context',default='exclusive serial window; one numerical worker and one library thread')
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing result')
    import run
    import squared_gap
    import numpy as np
    import scipy
    import torch
    import mpmath
    from threadpoolctl import threadpool_info
    squared_gap.install(run)
    start = time.perf_counter()
    solver = run.Solver(args.dimension,args.seed,'squared_gap',12,1500,args.geometry,8128,1.)
    setup = time.perf_counter()-start
    original_record = solver.record
    diagnostics = 0.
    def record(*a, **kw):
        nonlocal diagnostics
        t0 = time.perf_counter()
        try:
            return original_record(*a, **kw)
        finally:
            diagnostics += time.perf_counter()-t0
    solver.record = record
    result = solver.run()
    elapsed = time.perf_counter()-start
    endpoint = run.high_precision_error(solver.blocks,solver.coef,solver.tb,solver.tc,dps=110)
    assert abs(endpoint['relative']-result['final_relative_error']) <= max(1e-20,1e-10*endpoint['relative'])
    result['zero_squared_gaps'] = sum(b.get('t') == 0. for b in solver.blocks if b['kind']=='pair')
    result['squared_gaps'] = [b.get('t',b['h']**2) for b in solver.blocks if b['kind']=='pair']
    result['guard_counts'] = solver.guard_counts
    result['independent_endpoint_check_110_digits'] = endpoint
    result['timing'] = dict(context=args.timing_context,setup_seconds=setup,
        end_to_end_seconds=elapsed,diagnostic_record_seconds=diagnostics,
        end_to_end_without_diagnostic_records_seconds=elapsed-diagnostics,
        definition='Solver construction through run return; includes setup, diagnostics, scans, fits and restarts; excludes imports, process startup, writing and independent 110-digit audit.')
    result['ablation'] = dict(method='T',coordinate='squared halfgap t=h^2',bounds=[0.,(.6*.45)**2],
        irreversible_completions=False,boundary_can_reopen=True,
        pairing='unchanged guarded nearest-singleton grouping from frozen NC',
        gaussian_gram='exact cancellation-free parity formula',
        quadrature_power=12,quadrature_seed=8128,max_fits=1500,
        outer_options=dict(ftol=1e-15,gtol=1e-12,maxls=40,maxiter=6000))
    paths=sorted((HERE/'source').glob('*.py'))+[HERE/'squared_gap.py',Path(__file__)]
    result['reproduction']=dict(command=sys.argv,python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,mpmath=mpmath.__version__,
        threadpools=threadpool_info(),torch_threads=torch.get_num_threads(),pid=os.getpid(),
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(run.plain(result),indent=2)+'\n')
    print(json.dumps(dict(d=args.dimension,g=args.geometry,seed=args.seed,error=result['final_relative_error'],
                         fits=result['fits'],seconds=elapsed,zeros=result['zero_squared_gaps'],stop=result['stop'])),flush=True)


if __name__=='__main__':
    main()
