#!/usr/bin/env python3
"""Step-10 toy runs (pre-registered settings): trajectories at s=1, 0.99 and the gtol sweep at s=1."""
import json
import math
from pathlib import Path
import time

import toy

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
GTOLS = [10 ** (-k / 2) for k in range(2, 29)]


def summary(r):
    last = r['optimizer_returns'][-1] if r['optimizer_returns'] else {}
    return dict(variance=r['variance'], seed=r['seed'], method=r['method'], ftol=r['ftol'], gtol=r['gtol'],
                relative_error=r['final']['relative_error'], iterations=r['iterations'],
                objective_evaluations=r['objective_evaluations'], completions=r['completion_count'],
                h=r['final']['h'], t=r['final'].get('t'), a=r['final']['a'], c=r['final']['c'], beta=r['final']['beta'],
                stop=r['stop'], last_return_message=last.get('message'), returns=r['optimizer_returns'],
                hit_half_gap_floor=r['hit_half_gap_floor'], t_zero_at_end=r['t_zero_at_end'],
                final_gradient_max_abs=r['trace'][-1]['gradient_max_abs'])


def main():
    t0 = time.perf_counter()
    traj = []
    for s in (1.0, .99):
        for seed in range(10):
            for method in ('NC', 'C', 'T'):
                r = toy.run_one(s, seed, method)
                traj.append(r)
                print(f'traj s={s} seed={seed} {method}: E={r["final"]["relative_error"]:.3e} it={r["iterations"]} '
                      f'h={r["final"]["h"]:.2e} completions={r["completion_count"]} stop={r["stop"][:40]}', flush=True)
    (OUT / 'trajectories.json').write_text(json.dumps(traj) + '\n')
    sweep = []
    for g in GTOLS:
        for seed in range(10):
            for method in ('NC', 'C', 'T'):
                sweep.append(summary(toy.run_one(1.0, seed, method, ftol=0., gtol=g)))
        print(f'sweep gtol={g:.1e} done', flush=True)
    (OUT / 'gtol_sweep_toy.json').write_text(json.dumps(sweep, indent=1) + '\n')
    (OUT / 'toy_complete.json').write_text(json.dumps(dict(seconds=time.perf_counter() - t0, gtols=GTOLS)) + '\n')


if __name__ == '__main__':
    main()
