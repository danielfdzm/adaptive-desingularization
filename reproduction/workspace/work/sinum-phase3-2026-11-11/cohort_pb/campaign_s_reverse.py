#!/usr/bin/env python3
"""Extra single worker for the S arm (same run_case.py, same settings), processing S tasks in reverse order.

Runs beside the main campaign; both skip finished outputs, so at most the last task or two can coincide.
"""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
out = HERE / 'results'
tasks = [(d, g, s) for d in (5, 10, 20) for g in (1, 0) for s in range(34, 9, -1)]
for d, g, s in tasks:
    p = out / f'pb_d{d}_g{g}_s{s}_S.json'
    if p.exists():
        continue
    r = subprocess.run([sys.executable, str(HERE / 'run_case.py'), '--dimension', str(d), '--geometry', str(g),
                        '--seed', str(s), '--method', 'S', '--output', str(p)], capture_output=True, text=True)
    (out / (p.stem + '.log')).write_text(r.stdout + r.stderr)
    print(r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f'FAILED {p.name}', flush=True)
