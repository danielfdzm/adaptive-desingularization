"""Read-only numerical/coverage audit of the fixed validation campaigns."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    expected_counts={'unconstrained-validation':90,'unconstrained-log-validation':90,'unconstrained-pair-validation':60}
    parser.add_argument('--suites',nargs='+',choices=list(expected_counts),default=['unconstrained-validation'])
    parser.add_argument('--output',default='unconstrained-audit.json')
    args=parser.parse_args()
    results={}
    for dirname in args.suites:
        expected=expected_counts[dirname]
        root=HERE/dirname
        records=[]
        for path in sorted(root.glob('*.json')):
            r=json.loads(path.read_text())
            if isinstance(r,dict) and 'seed' in r and 'method' in r:records.append(r)
        valid=[r for r in records if r.get('error') is not None]
        quaddiff=max(abs(r['validation'][0]['error']-r['validation'][-1]['error']) for r in valid)
        energydiff=max(abs(r['validation'][0]['energy']-r['validation'][-1]['energy']) for r in valid)
        refdiff=max([r['reference_refinement_relative_H1'] for r in valid if r.get('reference_refinement_relative_H1') is not None] or [0.])
        events=[e for r in valid for e in r['events'] if e['accepted']]
        groups={}
        for r in records:groups.setdefault((r['problem'],r.get('capacity',2),r['seed']),[]).append(r)
        startmatch=all(len(rr)==3 and max(np.max(np.abs(np.array(r['initial_centers'])-np.array(rr[0]['initial_centers']))) for r in rr)==0 for rr in groups.values())
        item=dict(records=len(records),expected_records=expected,exceptions=len(records)-len(valid),all_methods_share_initial_centers=startmatch,
                  max_quadrature_error_change=quaddiff,max_quadrature_energy_change=energydiff,max_BVP_reference_change=refdiff,
                  accepted_completions=len(events),max_transfer_bound_excess=max([e['state_change']-e['analytic_bound'] for e in events] or [0.]),
                  state_bound_roundoff_tolerance=1e-10,
                  measured_state_changes_above_exact_arithmetic_bound=sum(e['state_change']>e['analytic_bound'] for e in events),
                  bound_note='Analytic bounds concern exact arithmetic; measured changes use floating-point basis evaluation. The stated absolute audit tolerance is not an interval enclosure.',
                  methods={})
        for method in ['ordinary','stable','adaptive']:
            rr=[r for r in valid if r['method']==method]
            item['methods'][method]=dict(runs=len(rr),abnormal_returns=sum('ABNORMAL' in r['status'] for r in rr),work_limits=sum(r['status']=='work_limit' for r in rr),max_kkt=max(r['kkt'] for r in rr),all_coefficients_unconstrained=all(r['budget'] is None and r['budget_ratio'] is None and r['coefficient_constraint']=='none' for r in rr))
        item['source_sha256']={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['run.py','multi.py','log_gap.py']}
        protocol=json.loads((root/'protocol.json').read_text())
        main_source='run.py' if dirname.endswith('pair-validation') else 'multi.py'
        item['source_hash_matches']=protocol['source_sha256']==item['source_sha256'][main_source]
        if main_source=='multi.py':
            item['source_hash_matches'] &= protocol['base_source_sha256']==item['source_sha256']['run.py']
        if dirname.endswith('log-validation'):
            coordinate=json.loads((root/'coordinate-protocol.json').read_text())
            item['source_hash_matches'] &= coordinate['source_sha256']==item['source_sha256']['log_gap.py']
        item['max_energy_gate_excess']=max([e['energy_change']-e.get('energy_allowance',5e-4) for e in events] or [0.])
        results[dirname]=item
        assert len(records)==expected and len(valid)==expected and startmatch
        assert all(x['all_coefficients_unconstrained'] for x in item['methods'].values())
        assert item['max_transfer_bound_excess']<item['state_bound_roundoff_tolerance']
        assert item['max_energy_gate_excess']<=1e-12 and item['source_hash_matches']
    (HERE/args.output).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
