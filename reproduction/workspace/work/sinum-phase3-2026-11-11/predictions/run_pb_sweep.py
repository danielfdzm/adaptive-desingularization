#!/usr/bin/env python3
"""Step-10 (a): gtol sweep with ftol=0 for the first-order PB start (d, geometry, seed) = (5, 0, 10).

NC, C and T run through ../cohort_pb/run_case.py (frozen solver, T chart unchanged); only the
L-BFGS-B tolerances and the fit limit (20000) are overridden. Each case is its own process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'cohort_pb' / 'run_case.py'
OUT = HERE / 'results' / 'pb_sweep'
GTOLS = [10 ** (-k / 2) for k in range(2, 29)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(k, m) for k in range(2, 29) for m in ('NC', 'C', 'T')]

    def one(task):
        k, m = task
        p = OUT / f'pb_d5_g0_s10_{m}_gtol{k:02d}.json'
        if p.exists():
            return f'{p.name} present'
        cmd = [sys.executable, str(RUN), '--dimension', '5', '--geometry', '0', '--seed', '10', '--method', m,
               '--max-fits', '20000', '--ftol', '0', '--gtol', repr(10 ** (-k / 2)), '--output', str(p),
               '--context', 'step-10 tolerance sweep; times not used']
        r = subprocess.run(cmd, capture_output=True, text=True)
        (OUT / (p.stem + '.log')).write_text(r.stdout + r.stderr)
        return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f'FAILED {p.name}: {r.stderr[-500:]}'

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            print(f.result(), flush=True)


if __name__ == '__main__':
    main()
