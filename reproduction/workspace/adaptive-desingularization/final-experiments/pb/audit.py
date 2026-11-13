#!/usr/bin/env python3
"""Structural, matched-start, budget, and source-provenance audit."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from config import configuration,start_centers

def audit(directory):
    directory=Path(directory)
    p=json.loads((directory/'provenance.json').read_text())
    expected={(d,g,s,m) for d in p['dimensions'] for g in p['geometries']
              for s in range(*p['seeds']) for m in p['methods']}
    rows=[json.loads(f.read_text()) for f in sorted(directory.glob('d*_g*_seed*_*.json'))]
    actual={(r['dimension'],r['geometry_id'],r['seed'],r['method']) for r in rows}
    checks=[]
    for r in rows:
        d,g,s,m=r['dimension'],r['geometry_id'],r['seed'],r['method']
        c=configuration(d,g)
        checks.append(dict(dimension=d,geometry_id=g,seed=s,method=m,
            start_reproduced=bool(np.array_equal(np.asarray(r['initial_centers']),start_centers(d,s,g))),
            target_definition_matches=bool(np.array_equal(np.asarray([b['c'] for b in r['target_blocks']]),np.asarray(c['target_centers'])) and
                np.array_equal(np.asarray([b['e'] for b in r['target_blocks'] if b['kind']=='pair']),np.asarray(c['target_directions'])) and
                np.array_equal(np.asarray(r['target_coefficients']),np.asarray(c['target_coefficients']))),
            quadrature_settings_match=(r['quadrature']['power']==p['power'] and
                r['quadrature']['seed']==p['quadrature_seed']+100*g+d and r['reaction_strength']==p['reaction_strength']),
            completion_records_match=sum(e['kind']=='completion' for e in r['events'])==r['completions'],
            capacity_preserved=sum(1 if b['kind']=='single' else 2 for b in r['blocks'])==10,
            budget_definition=r['C'] is None and c['C'] is None,
            final_budget_pass=r['C'] is None,
            completion_semantics=m=='adaptive' or r['completions']==0,
            fit_cap_pass=r['fits']<=p['max_fits'],
            finite_final_error=np.isfinite(r['final_relative_error']).item(),
            warm_start_semantics='accepted geometry states only'))
    source_root=Path(__file__).resolve().parent.parent
    hashes={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()==sha
            for name,sha in p['source_hashes'].items()}
    tests=('start_reproduced','target_definition_matches','quadrature_settings_match','completion_records_match','capacity_preserved','budget_definition','final_budget_pass',
           'completion_semantics','fit_cap_pass','finite_final_error')
    out=dict(run_count=len(rows),expected_run_count=len(expected),
             missing=[list(x) for x in sorted(expected-actual)],
             extra=[list(x) for x in sorted(actual-expected)],source_checks=hashes,checks=checks)
    out['passed']=actual==expected and len(rows)==len(expected) and all(hashes.values()) and all(
        all(row[t] for t in tests) for row in checks)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    args=parser.parse_args();out=audit(args.results)
    (args.results/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
    if not out['passed']:raise SystemExit(1)
