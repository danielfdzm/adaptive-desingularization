#!/usr/bin/env python3
import json,argparse
import run as r
import run_raw_v2 as raw
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','holdout']);phase=p.parse_args().phase
    if phase=='pilot':cases=[dict(a=a,A=A,lam=lam,seed=seed) for a,A,lam in r.PROTO['pilot_cases'] if a<=.5 for seed in r.PROTO['pilot_seeds']]
    else:cases=json.loads((r.HERE/'holdout_protocol.json').read_text())['cases']
    rows=[]
    for case in cases:
        row=raw.one(case['a'],case['A'],case['lam'],case['seed'],'S_vp',phase);rows.append(row)
        (r.HERE/f'{phase}_raw_v2_results.json').write_text(json.dumps(rows,indent=2))
        print(phase,case,'gap',row['energy_gap'],'h',row['h'],'calls',row['calls'],'rejected',len(row['failures']),row['status'],flush=True)
if __name__=='__main__':main()
