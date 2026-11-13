#!/usr/bin/env python3
"""Predeclared 30-start serial T cohort; historical NC/C retain provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    out=HERE/'results';out.mkdir(exist_ok=True)
    tasks=[dict(dimension=d,geometry=g,seed=s) for d in (5,10,20) for g in (0,1) for s in range(10,15)]
    paths=sorted((HERE/'source').glob('*.py'))+[HERE/'squared_gap.py',HERE/'run_case.py',Path(__file__)]
    manifest=dict(tasks=tasks,jobs=1,threads_per_worker=1,
        selection='All 30 Table 2 first-order PB starts, fixed before outcomes',
        protocol='Original NC grouping, coefficients, target, QMC quadrature, tolerances and budgets; t in [0,(0.6 sigma)^2]; no irreversible transfer',
        source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    m=out/'manifest.json'
    if m.exists():
        assert args.resume and json.loads(m.read_text())==manifest
    else:m.write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter()
    for i,t in enumerate(tasks,1):
        dest=out/f"pb_d{t['dimension']}_g{t['geometry']}_s{t['seed']}_squared_gap.json"
        if dest.exists():continue
        cmd=[sys.executable,str(HERE/'run_case.py')]
        for k,v in t.items():cmd.extend(['--'+k,str(v)])
        cmd.extend(['--output',str(dest)])
        result=subprocess.run(cmd,capture_output=True,text=True)
        dest.with_suffix('.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        print(f'{i}/30 '+result.stdout.strip(),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(count=30,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
