#!/usr/bin/env python3
"""Summarize all supplementary controls independently of main recovery cohorts."""
import collections, hashlib, json, statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tasks=json.loads((HERE/'tasks.json').read_text())
rows=[];comparisons=[];input_hashes={'tasks.json':sha(HERE/'tasks.json'),'source-manifest.json':sha(HERE/'source-manifest.json')}
for t in tasks:
    path=HERE/'accuracy'/(t['id']+'.json')
    if not path.exists():continue
    r=json.loads(path.read_text());oldpath=HERE/t['historical'];old=json.loads(oldpath.read_text())
    assert sha(oldpath)==t['historical_sha256']
    assert r['supplementary_task']==t
    input_hashes[str(path.relative_to(HERE))]=sha(path)
    input_hashes[t['historical']]=sha(oldpath)
    fields=['blocks','coefficients','fits','completions','final_relative_error','stop']
    strip=lambda items:[{k:v for k,v in item.items() if k not in ('wall','cancellation')} for item in items]
    checks={k:r[k]==old[k] for k in fields}
    checks['trace_excluding_time']=strip(r['trace'])==strip(old['trace'])
    checks['events_excluding_cancellation']=strip(r['events'])==strip(old['events'])
    comparisons.append(dict(id=t['id'],exact=all(checks.values()),checks=checks,
                            historical_error=old['final_relative_error'],new_error=r['final_relative_error']))
    rows.append(r)
expected=dict(collections.Counter(t['cohort'] for t in tasks))
found=dict(collections.Counter(r['supplementary_task']['cohort'] for r in rows))
manifest=json.loads((HERE/'source-manifest.json').read_text())
sources_match=all(sha(HERE/e['destination'])==e['sha256'] for e in manifest)
all_run_sources_match=all(sha(HERE/p)==h for r in rows for p,h in r['reproduction']['source_hashes'].items())
transfer=[v for r in rows for v in r['independent_transfer_checks']]
groups={}
for r in rows:
    t=r['supplementary_task'];key=f"{t['cohort']}:{t['case']}:{t['method']}"
    if t['energy_factor'] is not None:key+=f":factor{t['energy_factor']:g}"
    groups.setdefault(key,[]).append(r)
groups={k:dict(n=len(rr),median_recorded_error=statistics.median(r['final_relative_error'] for r in rr),
        median_independent_H1_error=statistics.median(r['independent_endpoint_check']['relative_H1'] for r in rr),
        median_fits=statistics.median(r['fits'] for r in rr),completions=sum(r['completions'] for r in rr),
        transfer_rejections=sum(r['transfer_rejections'] for r in rr),work_limits=sum(r['stop']=='work_limit' for r in rr)) for k,rr in groups.items()}
checks=dict(all_source_manifest_hashes_match=sources_match,all_run_source_hashes_match=all_run_sources_match,
    all_historical_results_reproduced=all(c['exact'] for c in comparisons),
    all_endpoint_checks_present=all('independent_endpoint_check' in r for r in rows),
    all_forcing_refined_errors_reproduced=all(abs(r['independent_endpoint_check']['relative_H1']-r['independent_endpoint_check']['historical_relative_H1_direct_refined'])<=5e-15*r['independent_endpoint_check']['historical_relative_H1_direct_refined'] for r in rows if r['supplementary_task']['snapshot']),
    all_transfer_state_bounds_pass=all(t['state_bound_pass'] and t['state_tolerance_pass'] for t in transfer),
    all_refined_transfer_gates_within_raw_allowance=all(t['refined_gate_order120_halfwidth5']<=t['allowance'] for t in transfer),
    all_refined_transfer_gates_within_allowance_and_numerical_tolerance=all(t['refined_gate_within_allowance_and_numerical_tolerance'] for t in transfer),
    all_threadpools_single_threaded=all(p['num_threads']==1 for r in rows for p in r['reproduction']['threadpools']),
    all_torch_single_threaded=all(r['reproduction']['torch_threads']==1 for r in rows))
summary=dict(complete=len(rows)==len(tasks),expected_count=len(tasks),found_count=len(rows),
    expected_by_cohort=expected,found_by_cohort=found,source_manifest='source-manifest.json',external_dependencies='external-dependencies.json',
    summary_generator_sha256=sha(Path(__file__)),input_hashes=input_hashes,checks=checks,
    accepted_transfers_checked=len(transfer),guard_totals={k:sum(r['guard_counts'][k] for r in rows) for k in ['tested','eligible','unresolved','nonfinite']},
    directional_guard='C.4 is inapplicable: first-order pair solvers have no directional joins.',
    timing='Accuracy-only shared host; no controlled runtime or speedup claim uses this campaign.',
    scope='Additional116 NC/C controls, separate from main258 accuracy cohort. Historical S and squared-gap/closure controls are not rerun.',
    quadrature_caution='Independent refined energy and forcing-grid comparisons are numerical checks, not continuum certificates; compact-source forcing pilot remains underresolved.',
    maximum_analytic_endpoint_recomputation_discrepancy=max((r['independent_endpoint_check'].get('discrepancy_from_recorded',0.) for r in rows),default=0.),
    forcing_refined_comparison=dict(relative_tolerance=5e-15,reason='The same direct-grid sum is multiplied by dx**2 in this wrapper and dx then dx in the historical validator, allowing last-bit floating-point differences.',
        maximum_absolute_discrepancy=max((abs(r['independent_endpoint_check']['relative_H1']-r['independent_endpoint_check']['historical_relative_H1_direct_refined']) for r in rows if r['supplementary_task']['snapshot']),default=0.)),
    groups=groups,historical_comparison=comparisons)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['complete','expected_count','found_count','found_by_cohort','checks','guard_totals','accepted_transfers_checked']},indent=2))
