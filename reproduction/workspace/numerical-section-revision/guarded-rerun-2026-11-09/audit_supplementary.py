#!/usr/bin/env python3
"""Verify saved supplementary controls without running numerical experiments."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

EXPECTED = {'recovery':20, 'mismatch001':20, 'mismatch005':20, 'allowance':20,
            'pilot-smoke':6, 'pilot-smooth':18, 'pilot-multi6':12}
REQUIRED_CHECKS = {'all_source_manifest_hashes_match', 'all_run_source_hashes_match',
    'all_historical_results_reproduced', 'all_endpoint_checks_present',
    'all_forcing_refined_errors_reproduced', 'all_transfer_state_bounds_pass',
    'all_refined_transfer_gates_within_allowance_and_numerical_tolerance',
    'all_threadpools_single_threaded', 'all_torch_single_threaded'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root, require_complete=True):
    root=Path(root)
    hashes={}
    def read(relative):
        path=root/relative
        hashes[str(relative)]=sha(path)
        return json.loads(path.read_text())
    summary=read('summary.json')
    tasks=read('tasks.json')
    assert len(tasks)==116 and len({t['id'] for t in tasks})==116
    assert dict(Counter(t['cohort'] for t in tasks))==EXPECTED
    assert summary['expected_count']==116 and summary['expected_by_cohort']==EXPECTED
    assert summary['summary_generator_sha256']==sha(root/'summarize.py'), 'Stale supplementary summary source'
    hashes['summarize.py']=summary['summary_generator_sha256']
    assert REQUIRED_CHECKS<=summary['checks'].keys()
    assert all(summary['checks'][k] for k in REQUIRED_CHECKS), summary['checks']
    if require_complete:
        assert summary['complete'] is True and summary['found_count']==116
        assert summary['found_by_cohort']==EXPECTED
    for name,digest in summary['input_hashes'].items():
        assert sha(root/name)==digest, ('Stale supplementary summary input', name)
        hashes[name]=digest
    for record in read('source-manifest.json'):
        name=record['destination']
        assert sha(root/name)==record['sha256'], ('Changed supplementary frozen dependency', name)
        hashes[name]=record['sha256']
    # The runner is deliberately self-contained. Additional external snapshots
    # would need an explicit review and packaging rule before release.
    assert read('external-dependencies.json')==[]
    comparisons={r['id']:r for r in summary['historical_comparison']}
    assert len(comparisons)==summary['found_count']
    by_cohort=Counter()
    transfers=0
    for task in tasks:
        relative='accuracy/'+task['id']+'.json'
        if relative not in summary['input_hashes']:
            assert not require_complete
            continue
        row=read(relative)
        assert row['supplementary_task']==task
        assert row['reproduction']['tasks_sha256']==hashes['tasks.json']
        assert 'accuracy' in row['timing']['context'] and 'not controlled' in row['timing']['context']
        assert math.isfinite(row['independent_endpoint_check']['relative_H1'])
        assert row['reproduction']['torch_threads']==1
        assert all(p['num_threads']==1 for p in row['reproduction']['threadpools'])
        for name,digest in row['reproduction']['source_hashes'].items():
            assert sha(root/name)==digest, ('Changed executed supplementary source', name)
            hashes[name]=digest
        old=read(task['historical'])
        assert hashes[task['historical']]==task['historical_sha256']
        comparison=comparisons[task['id']]
        assert comparison['exact'] and all(comparison['checks'].values())
        for field in ('blocks','coefficients','fits','completions','final_relative_error','stop'):
            assert row[field]==old[field], ('Supplementary historical mismatch', task['id'], field)
        strip=lambda items:[{k:v for k,v in item.items() if k not in ('wall','cancellation')} for item in items]
        assert strip(row['trace'])==strip(old['trace'])
        assert strip(row['events'])==strip(old['events'])
        for transfer in row['independent_transfer_checks']:
            assert transfer['state_bound_pass'] and transfer['state_tolerance_pass']
            assert transfer['refined_gate_within_allowance_and_numerical_tolerance']
            assert transfer['refined_gate_order120_halfwidth5']<=transfer['allowance']+transfer['quadrature_numerical_tolerance']
            transfers+=1
        by_cohort[task['cohort']]+=1
    assert dict(by_cohort)==summary['found_by_cohort']
    assert transfers==summary['accepted_transfers_checked']
    return summary,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    summary,hashes=inspect(Path(__file__).resolve().parent/'supplementary-2d',not args.partial)
    print(json.dumps(dict(complete=summary['complete'],expected=116,found=summary['found_count'],
                         verified_files=len(hashes),cohorts=summary['found_by_cohort']),indent=2))


if __name__=='__main__': main()
