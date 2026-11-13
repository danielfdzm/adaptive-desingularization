#!/usr/bin/env python3
"""Recompute the 40 independent-cohort historical comparisons from saved files."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest=json.loads((HERE/'historical-baselines/manifest.json').read_text())
    fields=('final_blocks','final_coefficients','initial_centers','target',
            'final_relative_error','fits','joins','completions')
    comparisons=[]
    for order in (2,3):
        for seed in range(900,910):
            original=manifest[f'order{order}_seed{seed}_standard_restart.json']['source']
            for method in ('adaptive','stable'):
                name=f'order{order}_seed{seed}_{method}.json'
                oldpath=WORKSPACE/original.replace('_standard_restart.json',f'_{method}.json')
                newpath=HERE/'independent/accuracy'/name
                old=json.loads(oldpath.read_text());new=json.loads(newpath.read_text())
                comparisons.append(dict(name=name,equal={key:new[key]==old[key] for key in fields},
                    new_sha256=hashlib.sha256(newpath.read_bytes()).hexdigest(),
                    historical_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                    historical_path=str(oldpath.relative_to(WORKSPACE))))
    result=dict(all_40_endpoints_exactly_reproduced=all(all(row['equal'].values()) for row in comparisons),comparisons=comparisons)
    (HERE/'independent/historical-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(comparisons=len(comparisons),all_exact=result['all_40_endpoints_exactly_reproduced']),indent=2))


if __name__=='__main__':main()
