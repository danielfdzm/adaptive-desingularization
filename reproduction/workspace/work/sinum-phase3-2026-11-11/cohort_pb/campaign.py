#!/usr/bin/env python3
"""Pre-registered first-order PB cohort: d=5,10,20; geometries 0,1; seeds 10-34; S, NC, C, T.

Each case runs in its own process (T monkeypatches the solver module).
Existing outputs are kept; --resume requires an identical manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SQDIR = ROOT / 'numerical-section-revision' / 'sinum-revision-2026-11-10' / 'squared-gap'


def tasks():
    out = []
    for method in ('T', 'NC', 'C', 'S'):          # amendment 1: method runs first, the slow S baseline last
        for d in (20, 10, 5):
            for g in (0, 1):
                for s in range(10, 35):
                    out.append(dict(dimension=d, geometry=g, seed=s, method=method))
    return out


def ident(t):
    return f"pb_d{t['dimension']}_g{t['geometry']}_s{t['seed']}_{t['method']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    tt = tasks()
    sources = sorted((SQDIR / 'source').glob('*.py')) + [SQDIR / 'squared_gap.py', HERE / 'run_case.py']
    manifest = dict(tasks=tt, jobs=args.jobs, prereg_sha256=(ROOT / 'work/sinum-phase3-2026-11-11/prereg/prereg.sha256').read_text(),
                    timing_context='accuracy campaign with several concurrent workers; wall times not used for comparisons',
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    mf = out / 'manifest.json'
    if mf.exists():
        old = json.loads(mf.read_text())
        same_sources = all(old['source_sha256'].get(k) == v for k, v in manifest['source_sha256'].items())
        same_tasks = sorted(map(json.dumps, old['tasks'])) == sorted(map(json.dumps, manifest['tasks']))
        if not args.resume or not same_sources or not same_tasks:
            raise SystemExit('Campaign exists with different solver sources or tasks; refusing to resume.')
        launches = old.get('launches', [dict(jobs=old['jobs'], order='S, NC, C, T')])
        launches.append(dict(jobs=args.jobs, order='T, NC, C, S', time=time.strftime('%Y-%m-%d %H:%M:%S'),
                             reason='coordination with the parallel Phase-1 runs: at most four Phase-3 workers; '
                                    'method runs before the S baseline. Solver and run_case sources unchanged.'))
        old['launches'] = launches
        mf.write_text(json.dumps(old, indent=2) + '\n')
    else:
        mf.write_text(json.dumps(manifest, indent=2) + '\n')

    def one(t):
        p = out / (ident(t) + '.json')
        if p.exists():
            return dict(id=ident(t), status='present')
        cmd = [sys.executable, str(HERE / 'run_case.py')]
        for k, v in t.items():
            cmd += ['--' + k, str(v)]
        cmd += ['--output', str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / (ident(t) + '.log')).write_text(r.stdout + r.stderr)
        if r.returncode:
            return dict(id=ident(t), status='FAILED', stderr=r.stderr[-1500:])
        return json.loads(r.stdout.splitlines()[-1])

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, t) for t in tt]
        for i, f in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=i, total=len(tt), elapsed=round(time.perf_counter() - start), **f.result())), flush=True)
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest['source_sha256'][str(p.relative_to(ROOT))]:
            raise RuntimeError(f'source changed during campaign: {p}')
    (out / 'complete.json').write_text(json.dumps(dict(count=len(tt), elapsed_seconds=time.perf_counter() - start), indent=2) + '\n')


if __name__ == '__main__':
    main()
