#!/usr/bin/env python3
"""Aggregate declared PB dataset audits and independent validation outcomes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from reproduce import studies
HERE=Path(__file__).resolve().parent

def main(root):
    rows=[];strength=[];transfer_count=check_count=actual=expected=0
    for name,dims,geoms,seeds,methods,power,qseed,reaction in studies():
        directory=root/name
        count=len(dims)*len(geoms)*(seeds[1]-seeds[0])*len(methods)
        expected+=count
        n=len(list(directory.glob('d*_g*_seed*_*.json')));actual+=n
        audit=json.loads((directory/'audit.json').read_text()) if (directory/'audit.json').exists() else {}
        valid=json.loads((directory/'validation.json').read_text()) if (directory/'validation.json').exists() else {}
        transfers=[t for r in valid.get('runs',[]) for t in r['transfers']]
        increments=[q for t in transfers for q in t['independent_increments']]
        transfer_count+=len(transfers);check_count+=len(increments)
        rows.append(dict(study=name,expected_runs=count,actual_runs=n,audit_passed=audit.get('passed',False),
                         validation_passed=valid.get('passed',False),transfer_count=len(transfers),
                         independent_energy_check_count=len(increments),
                         minimum_independent_allowance_margin=min((x['allowance_margin'] for x in increments),default=None),
                         maximum_state_change_to_bound_ratio=max((t['state_change_norm']/t['state_error_bound'] for t in transfers if t['state_error_bound']>0),default=None)))
        if name in ('results','calibrated'):
            strength.extend(dict(study=name,**g) for g in valid.get('groups',[]))
    out=dict(expected_runs=expected,actual_runs=actual,transfer_count=transfer_count,
             independent_energy_check_count=check_count,studies=rows,nonlinearity=strength,
             rigorous_integration_certificate=False)
    out['passed']=actual==expected and all(r['actual_runs']==r['expected_runs'] and r['audit_passed'] and r['validation_passed'] for r in rows)
    (root/'validation-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='nonlinearity'},indent=2))
    return out['passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=HERE)
    if not main(parser.parse_args().root):raise SystemExit(1)
