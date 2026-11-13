#!/usr/bin/env python3
"""Amendment 3: C_d with the exact parity evaluator (C_d-exact) on all 150 first-order PB starts.

Runs the Phase-1 evaluator-control runner (first_order_runner.py --mode pb_NC_analytic) once per start,
each in its own process. Existing outputs are kept; the runner hash is recorded and re-checked at the end.
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
RUNNER = ROOT / 'numerical-section-revision' / 'phase1-2026-11-11' / 'first_order_runner.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    out = HERE / 'results_cd_exact'
    out.mkdir(exist_ok=True)
    sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    mf = out / 'manifest.json'
    tasks = [(d, g, s) for d in (20, 10, 5) for g in (0, 1) for s in range(10, 35)]
    if mf.exists():
        old = json.loads(mf.read_text())
        if old['runner_sha256'] != sha:
            raise SystemExit('runner changed; refusing to resume')
    else:
        mf.write_text(json.dumps(dict(runner=str(RUNNER.relative_to(ROOT)), runner_sha256=sha, mode='pb_Cd_analytic',
                                      tasks=tasks, amendment='prereg/AMENDMENT-3.md'), indent=2) + '\n')

    def one(t):
        d, g, s = t
        p = out / f'pb_d{d}_g{g}_s{s}_Cdx.json'
        if p.exists():
            return f'{p.name} present'
        r = subprocess.run([sys.executable, str(RUNNER), '--mode', 'pb_Cd_analytic', '--dimension', str(d),
                            '--geometry', str(g), '--seed', str(s), '--output', str(p)], capture_output=True, text=True)
        (out / (p.stem + '.log')).write_text(r.stdout + r.stderr)
        return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f'FAILED {p.name}: {r.stderr[-400:]}'

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, f in enumerate(as_completed([pool.submit(one, t) for t in tasks]), 1):
            print(f'{i}/150 {round(time.perf_counter() - start)}s {f.result()}', flush=True)
    if hashlib.sha256(RUNNER.read_bytes()).hexdigest() != sha:
        raise SystemExit('runner changed during the campaign')
    (out / 'complete.json').write_text(json.dumps(dict(count=len(tasks), seconds=time.perf_counter() - start)) + '\n')


if __name__ == '__main__':
    main()
