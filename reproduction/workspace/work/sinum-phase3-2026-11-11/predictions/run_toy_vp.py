#!/usr/bin/env python3
"""Variable-projection toy (added after the joint-protocol sweep): trajectories at s=1, 0.99 and the gtol sweep at s=1."""
import json
from pathlib import Path
import time
import toy_vp

OUT = Path(__file__).resolve().parent / 'results'
GTOLS = [10 ** (-k / 2) for k in range(2, 29)]
t0 = time.perf_counter()
traj = [toy_vp.run_one(s, seed, m) for s in (1.0, .99) for seed in range(10) for m in ('NC', 'C', 'T')]
(OUT / 'vp_trajectories.json').write_text(json.dumps(traj) + '\n')
sweep = []
for g in GTOLS:
    for seed in range(10):
        for m in ('NC', 'C', 'T'):
            r = toy_vp.run_one(1.0, seed, m, ftol=0., gtol=g)
            r.pop('trace'); sweep.append(r)
    print(f'vp sweep gtol={g:.1e} done', flush=True)
(OUT / 'vp_gtol_sweep.json').write_text(json.dumps(sweep) + '\n')
(OUT / 'vp_complete.json').write_text(json.dumps(dict(seconds=time.perf_counter() - t0)) + '\n')
