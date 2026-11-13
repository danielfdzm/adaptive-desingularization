#!/usr/bin/env python3
"""Fixed matched cohorts, with separate shared-host accuracy and serial timing runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def tasks(mode):
    result=[]
    for d in (5,10,20):
        starts=[(g,s) for g in (0,1) for s in range(10,15)] if mode=='accuracy' else [(0,10),(0,11),(1,10)]
        for g,s in starts:
            for m in ('stable','adaptive'):
                result.append(dict(family='pb',dimension=d,geometry=g,seed=s,method=m))
    for d in (10,20):
        for s in (10,11,12):
            for m in ('stable','adaptive'):
                result.append(dict(family='linear',dimension=d,geometry=0,seed=s,method=m))
    return result

def ident(t):
    return f"{t['family']}_d{t['dimension']}_g{t['geometry']}_s{t['seed']}_{t['method']}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['accuracy','benchmark'],required=True)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.mode=='benchmark' and args.jobs!=1:
        raise SystemExit('Benchmark requires --jobs 1 and coordination for an otherwise idle host.')
    out=HERE/args.mode;out.mkdir(exist_ok=True)
    tt=tasks(args.mode)
    manifest=dict(mode=args.mode,jobs=args.jobs,tasks=tt,selection='declared before new numerical outcomes',
        historical_standard='Retained only as historical accuracy records; no new S timing or guarded-method claim.',
        timing_context=('one numerical worker; exclusive host window coordinated by parent' if args.mode=='benchmark'
                        else 'shared-host accuracy campaign; not used for speed comparisons'),
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((HERE/'source').rglob('*.py'))+
                       [HERE/'pair_guard.py',HERE/'run_case.py',Path(__file__)]})
    mf=out/'manifest.json'
    if mf.exists():
        if not args.resume:raise SystemExit('Campaign exists; use --resume.')
        previous=json.loads(mf.read_text())
        assert previous==manifest,'Campaign manifest changed'
    else:mf.write_text(json.dumps(manifest,indent=2)+'\n')
    def one(t):
        p=out/(ident(t)+'.json')
        if p.exists():return dict(id=ident(t),status='already present')
        cmd=[sys.executable,str(HERE/'run_case.py')]
        for k,v in t.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
        cmd.extend(['--output',str(p),'--timing-context',manifest['timing_context']])
        r=subprocess.run(cmd,capture_output=True,text=True)
        (out/(ident(t)+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError(f'{ident(t)} failed: {r.stderr[-2000:]}')
        return json.loads(r.stdout.splitlines()[-1])
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(one,t) for t in tt]
        for i,f in enumerate(as_completed(futures),1):
            print(json.dumps(dict(completed=i,total=len(tt),**f.result())),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(count=len(tt),elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')

if __name__=='__main__':main()
