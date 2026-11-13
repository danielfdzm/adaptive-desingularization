#!/usr/bin/env python3
"""Compare reoptimized states under changed training rules using exact H1 norms."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from kernels import high_precision_error
HERE=Path(__file__).resolve().parent

def minimum_margin(run):
    return min((e['eps_j']-e['gate'] for e in run['events'] if e['kind']=='completion'),default=None)

def independent_margin(directory,run):
    path=directory/'validation.json'
    if not path.exists():return None
    key=f"d{run['dimension']}_g{run['geometry_id']}_seed{run['seed']}_{run['method']}"
    data=json.loads(path.read_text())
    entries=[r for r in data['runs'] if r['id']==key]
    return min((q['allowance_margin'] for r in entries for t in r['transfers'] for q in t['independent_increments']),default=None)

def main(root=HERE):
    comparisons=[]
    for directory in sorted(root.glob('retrain_*')):
        if not directory.is_dir():continue
        baseline='calibrated' if 'calibrated' in directory.name else 'results'
        for path in sorted(directory.glob('d*_g*_seed*_*.json')):
            a=json.loads((root/baseline/path.name).read_text());b=json.loads(path.read_text())
            norm=high_precision_error(a['blocks'],a['coefficients'],b['blocks'],b['coefficients'])['norm']
            comparisons.append(dict(study=directory.name,dimension=b['dimension'],method=b['method'],
                geometry_id=b['geometry_id'],seed=b['seed'],reaction_strength=b['reaction_strength'],
                baseline_error=a['final_relative_error'],retrained_error=b['final_relative_error'],
                state_difference_relative_to_target=norm/b['target_norm'],
                baseline_completions=a['completions'],retrained_completions=b['completions'],
                baseline_power=a['quadrature']['power'],retrained_power=b['quadrature']['power'],
                baseline_scramble=a['quadrature']['seed'],retrained_scramble=b['quadrature']['seed'],
                baseline_stop=a['stop'],retrained_stop=b['stop'],
                baseline_min_training_gate_margin=minimum_margin(a),retrained_min_training_gate_margin=minimum_margin(b),
                baseline_min_independent_gate_margin=independent_margin(root/baseline,a),
                retrained_min_independent_gate_margin=independent_margin(directory,b),
                baseline_projected_gradient=a['projected_gradient_max_abs'],
                retrained_projected_gradient=b['projected_gradient_max_abs']))
    out=dict(comparison_count=len(comparisons),comparisons=comparisons,
             interpretation='Differences include optimizer path sensitivity; matched manufactured loads preserve the target for every training rule.')
    (root/'training-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE)
    main(parser.parse_args().root)
