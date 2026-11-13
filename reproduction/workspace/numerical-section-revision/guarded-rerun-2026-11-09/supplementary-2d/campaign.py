#!/usr/bin/env python3
"""Retain every prespecified supplementary outcome; independent subprocesses."""
import argparse, concurrent.futures, json, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=2);args=ap.parse_args()
assert 1<=args.jobs<=2
tasks=json.loads((HERE/'tasks.json').read_text());out=HERE/'accuracy';out.mkdir(exist_ok=True)
pending=[t for t in tasks if not (out/(t['id']+'.json')).exists()]
def run(t):
    p=subprocess.run([sys.executable,str(HERE/'run_case.py'),'--id',t['id'],'--output',str(out/(t['id']+'.json'))],capture_output=True,text=True)
    if p.returncode:
        (out/(t['id']+'.error.txt')).write_text(p.stdout+'\n'+p.stderr)
        raise RuntimeError(t['id']+': '+p.stderr[-2000:])
    return p.stdout
with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    futures={pool.submit(run,t):t for t in pending}
    for f in concurrent.futures.as_completed(futures):print(f.result(),end='',flush=True)
assert all((out/(t['id']+'.json')).exists() for t in tasks)
(out/'complete.json').write_text(json.dumps(dict(complete=True,count=len(tasks),finished_unix=time.time(),context='accuracy-only shared host'),indent=2)+'\n')
