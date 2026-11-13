#!/usr/bin/env python3
"""Audit fixed cohort coverage and build manuscript summaries from saved records.

Historical S endpoints remain historical. Accuracy-campaign wall times are
never promoted to controlled runtimes. No optimizer or reference solver runs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from audit_supplementary import inspect as inspect_supplementary

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
METHODS={'standard':'S','standard_restart':'S','stable':'NC','ablation':'NC','adaptive':'C'}
DERIVED_MATCH_FIELDS=('final_blocks','final_coefficients','events','fits','iterations','joins','completions','final_relative_error','guard_counts','stop')
TABLE_STUDIES=['linear1d','pb_d5','pb_d10','pb_d20','second1d','higher_d5','linear_d10','linear_d20']
LABELS={
    'linear1d':r'Linear, $d=1$, $r\le2$',
    'pb_d5':r'PB, $d=5$, $r=1$', 'pb_d10':r'PB, $d=10$, $r=1$',
    'pb_d20':r'PB, $d=20$, $r=1$', 'second1d':r'PB, $d=1$, $r=2$',
    'higher_d5':r'PB, $d=5$, $r\le3$',
    'linear_d10':r'Linear, $d=10$, span', 'linear_d20':r'Linear, $d=20$, span',
    'independent_r2':r'Independent, $r=2$', 'independent_r3':r'Independent, $r=3$',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sci(value,decimals=1):
    if value is None:
        return '--'
    mantissa,exponent=f'{value:.{decimals}e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def numeric(value):
    return '--' if value is None else f'{value:g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='Explicitly emit incomplete live snapshots with .partial filenames.')
    args=parser.parse_args()
    inputs={}
    missing=[]
    rows=[]
    runtime_rows=[]
    raw={}
    pairing=[]
    source_checks={}
    historical=HERE/'historical-baselines'

    def read(path,required=True):
        path=Path(path)
        if not path.exists():
            if required: missing.append(str(path.relative_to(HERE)))
            return None
        try:
            value=json.loads(path.read_text())
        except json.JSONDecodeError:
            if not args.partial: raise
            missing.append(str(path.relative_to(HERE))+' (incomplete JSON)')
            return None
        inputs[str(path.relative_to(HERE))]=sha(path)
        return value

    manifest=read(historical/'manifest.json')
    assert manifest is not None
    assert len(manifest)==77,('historical baseline file count',len(manifest))
    for name,info in manifest.items():
        path=historical/name
        assert path.exists(),path
        assert sha(path)==info['sha256'],('historical copy hash',name)
        original=WORKSPACE/info['source']
        if original.exists():
            assert sha(original)==info['sha256'],('historical original hash',name)

    def add(path,study,geometry,seed,method,provenance,timing=False):
        row=read(path)
        if row is None: return
        for rel,expected in row.get('reproduction',{}).get('source_hashes',{}).items():
            source=HERE/'first-order'/rel
            if str(source) not in source_checks:
                source_checks[str(source)]=sha(source)
            assert source_checks[str(source)]==expected,('first-order frozen source',rel,path)
        assert METHODS[row.get('method',row.get('kind'))]==method,(path,'method')
        assert row['seed']==seed,(path,'seed')
        if 'geometry_id' in row: assert row['geometry_id']==geometry
        if isinstance(row.get('geometry'),int): assert row['geometry']==geometry
        error=row.get('final_relative_error',row.get('final_rel_err_quad'))
        assert isinstance(error,(int,float)) and math.isfinite(error) and error>=0,(path,'error')
        item=dict(study=study,geometry=geometry,seed=seed,method=method,
                  provenance=provenance,path=str(Path(path).relative_to(HERE)),
                  error=error,fits=row['fits'],iterations=row.get('iterations',row.get('lbfgs_iters')),
                  completions=row.get('completions'),joins=row.get('joins'),
                  stop=row['stop'],guard_counts=row.get('guard_counts',{}),
                  solver_variant='frozen guarded source' if 'fresh guarded' in provenance or timing else 'historical',
                  matches_frozen_accuracy=None,guarded_restart_returns=row.get('guarded_restart_returns',0),
                  timing_mode=row.get('timing_mode'),concurrent_workers=row.get('concurrent_workers'))
        if not timing and study in ('higher_d5','second1d') and provenance=='fresh guarded accuracy rerun':
            assert row.get('timing_mode')=='accuracy' and row.get('concurrent_workers') in (2,4),(path,'accuracy scheduling metadata')
        if timing:
            if study in ('higher_d5','second1d'):
                assert row.get('derived_solver_version')=='restart-safe-v1',(path,'missing derived solver version')
                derived=row.get('derived_solver_sha256',{})
                assert set(derived)=={'restart_safe.py'},(path,'missing wrapper hash')
                for rel,digest in derived.items():
                    source=HERE/'directional'/rel
                    assert sha(source)==digest,('controlled derived source changed',rel,path)
                    source_checks[str(source)]=digest
                matches=row.get('matches_frozen_accuracy',{})
                assert set(matches)==set(DERIVED_MATCH_FIELDS) and all(matches.values()),(path,'controlled/frozen endpoint mismatch')
                assert row.get('matches_frozen_accuracy_all') is True
                item.update(solver_variant=row['derived_solver_version'],matches_frozen_accuracy=True)
            if 'timing' in row:
                context=row['timing']['context']
                assert 'accuracy' not in context.lower() and 'shared-host' not in context.lower(),(path,context)
                seconds=row['timing']['end_to_end_seconds']
                scope=row['timing']['definition']
            elif 'timing_mode' in row:
                assert row['timing_mode']=='controlled' and row['concurrent_workers']==1,(path,'uncontrolled timing')
                seconds=row['end_to_end_seconds'];scope=row['timing_scope']
            else:
                assert row.get('benchmark') is True,(path,'missing controlled benchmark flag')
                seconds=row['end_to_end_seconds']
                protocol=read(Path(path).parent/'protocol.json')
                assert protocol and protocol['benchmark'] and protocol['workers']==1
                scope=protocol['timing_scope']
            assert math.isfinite(seconds) and seconds>=0
            item.update(seconds=seconds,timing_scope=scope)
            runtime_rows.append(item)
        else:
            rows.append(item)
        raw[('runtime' if timing else 'accuracy',study,geometry,seed,method)]=row

    legacy=read(historical/'linear1d-shared.json')
    assert len(legacy['runs'])==30
    seen=set()
    for row in legacy['runs']:
        method=METHODS[row['kind']];seed=row['seed'];key=(seed,method)
        assert key not in seen;seen.add(key)
        rows.append(dict(study='linear1d',geometry=0,seed=seed,method=method,
                         provenance='historical; previously matched rule rerun, not part of the new guarded campaign',
                         path='historical-baselines/linear1d-shared.json',error=row['final_rel_err_quad'],
                         fits=row['fits'],iterations=row['lbfgs_iters'],completions=None,joins=None,
                         stop=row['stop'],guard_counts={},solver_variant='historical',
                         matches_frozen_accuracy=None,guarded_restart_returns=0,timing_mode=None,concurrent_workers=None))
        raw[('accuracy','linear1d',0,seed,method)]=row
    assert seen=={(s,m) for s in range(10) for m in ('S','NC','C')}

    # First-order PB and full-span linear cohorts.
    for family,dimensions in [('pb',(5,10,20)),('linear',(10,20))]:
        for d in dimensions:
            study=f'{family}_d{d}'
            cases=[(g,s) for g in (0,1) for s in range(10,15)] if family=='pb' else [(0,s) for s in (10,11,12)]
            timed=[(0,10),(0,11),(1,10)] if family=='pb' else cases
            for g,s in cases:
                add(historical/f'{family}_d{d}_g{g}_s{s}_standard.json',study,g,s,'S','unchanged historical S endpoint')
                for name,method in [('stable','NC'),('adaptive','C')]:
                    add(HERE/f'first-order/accuracy/{family}_d{d}_g{g}_s{s}_{name}.json',study,g,s,method,'fresh guarded accuracy rerun')
            for g,s in timed:
                for name,method in [('stable','NC'),('adaptive','C')]:
                    add(HERE/f'first-order/benchmark/{family}_d{d}_g{g}_s{s}_{name}.json',study,g,s,method,'separate serial controlled benchmark',True)

    # Directional higher order and target-informed 1D second order.
    for problem,study,d,cases,timed in [
        ('higher','higher_d5',5,[(g,s) for g in (0,1) for s in range(10,15)],[(0,10),(0,11),(1,10)]),
        ('second1d','second1d',1,[(0,s) for s in range(40,50)],[(0,s) for s in (40,41,42)])]:
        for g,s in cases:
            add(historical/f'{problem}_d{d}_g{g}_seed{s}_standard.json',study,g,s,'S','unchanged historical S endpoint')
            for name,method in [('stable','NC'),('adaptive','C')]:
                add(HERE/f'directional/accuracy-{problem}/{problem}_d{d}_g{g}_seed{s}_{name}.json',study,g,s,method,'fresh guarded accuracy rerun')
        for g,s in timed:
            for name,method in [('stable','NC'),('adaptive','C')]:
                add(HERE/f'directional/timing-{problem}/{problem}_d{d}_g{g}_seed{s}_{name}.json',study,g,s,method,'separate serial controlled benchmark',True)

    # Independent confirmation keeps its original S baseline; NC/C are fresh.
    for order in (2,3):
        study=f'independent_r{order}'
        for seed in range(900,910):
            add(historical/f'order{order}_seed{seed}_standard_restart.json',study,0,seed,'S','unchanged historical S endpoint')
            for name,method in [('stable','NC'),('adaptive','C')]:
                add(HERE/f'independent/accuracy/order{order}_seed{seed}_{name}.json',study,0,seed,method,'fresh guarded accuracy rerun')
        for seed in (900,901,902):
            for name,method in [('stable','NC'),('adaptive','C')]:
                add(HERE/f'independent/benchmark/order{order}_seed{seed}_{name}.json',study,0,seed,method,'separate serial controlled benchmark',True)

    # Verify actual shared starts and targets wherever present; otherwise retain
    # explicit deterministic-constructor provenance instead of claiming equality.
    for category in ('accuracy','runtime'):
        cases=sorted({key[1:4] for key in raw if key[0]==category})
        for study,g,seed in cases:
            available={m:raw[(category,study,g,seed,m)] for m in ('S','NC','C') if (category,study,g,seed,m) in raw}
            if len(available)<2: continue
            comparable=[]
            for field in ('initial_centers','target_blocks','target_coefficients','quadrature','target','input_sha256'):
                values=[(m,r[field]) for m,r in available.items() if field in r]
                if len(values)>1:
                    first=values[0][1]
                    assert all(v==first for _,v in values),(category,study,g,seed,field)
                    comparable.append(dict(field=field,methods=[m for m,_ in values]))
            record=dict(category=category,study=study,geometry=g,seed=seed,checked=comparable)
            if study in ('second1d','higher_d5'):
                record['historical_S_evidence']='Same problem/geometry/seed/slots; problem constructors AST-identical to historical sources, with unchanged config/quadrature hashes. Historical S files do not contain explicit initial centers.'
            if study=='linear1d':
                record['evidence']='Shared-seed cohort and target metadata in one historical source; no new guarded rerun.'
            pairing.append(record)

    for key,record in raw.items():
        if key[0]!='runtime': continue
        accuracy_key=('accuracy',)+key[1:]
        if accuracy_key not in raw: continue
        same=[]
        for field in ('initial_centers','target_blocks','target_coefficients','quadrature','target','input_sha256'):
            if field in record and field in raw[accuracy_key]:
                assert record[field]==raw[accuracy_key][field],('benchmark/accuracy input mismatch',key,field)
                same.append(field)
        if key[1] in ('higher_d5','second1d'):
            problem,dimension=('higher',5) if key[1]=='higher_d5' else ('second1d',1)
            method_name='stable' if key[4]=='NC' else 'adaptive'
            accuracy_path=HERE/f'directional/accuracy-{problem}/{problem}_d{dimension}_g{key[2]}_seed{key[3]}_{method_name}.json'
            assert record.get('frozen_accuracy_sha256')==sha(accuracy_path),('derived benchmark frozen input hash',key)
            for field in DERIVED_MATCH_FIELDS:
                assert record[field]==raw[accuracy_key][field],('derived benchmark differs from frozen accuracy',key,field)
            same+=list(DERIVED_MATCH_FIELDS)
        pairing.append(dict(category='benchmark-versus-accuracy',study=key[1],geometry=key[2],seed=key[3],method=key[4],checked_fields=same))

    directional_provenance=read(HERE/'directional/implementation-provenance.json')
    assert directional_provenance['problem_constructor_ast_identical_to_published_sources']==dict(problem_higher=True,problem_second1d=True)
    launcher_history=read(HERE/'directional/launcher-history.json',False)
    read(HERE/'directional/resource-assessment.json',False)
    hardware=read(HERE/'hardware.json')
    if launcher_history:
        for event in launcher_history['events']:
            for path_key,hash_key in [('provenance_file','provenance_sha256'),('resumed_provenance_file','resumed_provenance_sha256')]:
                if path_key in event:
                    path=HERE/'directional'/event[path_key]
                    saved=read(path)
                    assert sha(path)==event[hash_key],('scheduling provenance changed',path)
                    assert saved['timing_mode']=='accuracy' and saved['jobs'] in (2,4)
                    for rel,digest in saved['source_sha256'].items():
                        assert sha(HERE/'directional/src'/rel)==digest
            for name,digest in event.get('completed_files_retained',event.get('completed_before_restart',{})).items():
                assert sha(HERE/'directional/accuracy-higher'/name)==digest,('completed outcome changed across scheduling restart',name)
    higher_workers={r['concurrent_workers'] for r in rows if r['study']=='higher_d5' and r['provenance']=='fresh guarded accuracy rerun'}
    if len(higher_workers)>1:
        assert launcher_history, 'Mixed accuracy worker counts require scheduling history'
    for name in ('accuracy-second1d','accuracy-higher','timing-second1d','timing-higher'):
        provenance=read(HERE/f'directional/{name}/provenance.json',False)
        if provenance:
            for rel,expected in provenance['source_sha256'].items():
                source=HERE/'directional/src'/rel
                assert sha(source)==expected,('directional frozen source',name,rel)
                source_checks[str(source)]=expected
            if name.startswith('timing-'):
                assert provenance.get('derived_solver_version')=='restart-safe-v1',(name,'derived provenance missing')
                derived=provenance.get('derived_solver_sha256',{})
                assert set(derived)=={'restart_safe.py'},(name,'derived provenance hash missing')
                for rel,digest in derived.items():
                    source=HERE/'directional'/rel
                    assert sha(source)==digest,('directional benchmark wrapper hash',rel)
                    source_checks[str(source)]=digest
    for name in ('accuracy','benchmark'):
        protocol=read(HERE/f'independent/{name}/protocol.json',False)
        if protocol:
            for rel,expected in protocol['source_sha256'].items():
                source=WORKSPACE/rel
                assert sha(source)==expected,('independent frozen source',name,rel)
                source_checks[str(source)]=expected

    def summarize(rr,runtime=False):
        if not rr: return None
        counts=Counter()
        for r in rr: counts.update(r['guard_counts'])
        out=dict(n=len(rr),median_error=statistics.median(r['error'] for r in rr),
                 maximum_error=max(r['error'] for r in rr),
                 successes_1e6=sum(r['error']<=1e-6 for r in rr),successes_1e10=sum(r['error']<=1e-10 for r in rr),
                 median_fits=statistics.median(r['fits'] for r in rr),
                 median_iterations=statistics.median(r['iterations'] for r in rr),
                 total_completions=sum(r['completions'] or 0 for r in rr),
                 total_joins=sum(r['joins'] or 0 for r in rr),guard_counts=dict(counts),
                 solver_variants=sorted({r['solver_variant'] for r in rr}),
                 recorded_concurrent_workers=sorted({r['concurrent_workers'] for r in rr if r['concurrent_workers'] is not None}),
                 guarded_restart_returns=sum(r['guarded_restart_returns'] for r in rr),
                 stopping_counts=dict(Counter(r['stop'] for r in rr)),
                 provenance=sorted({r['provenance'] for r in rr}))
        if runtime:
            out.update(median_seconds=statistics.median(r['seconds'] for r in rr),
                       minimum_seconds=min(r['seconds'] for r in rr),maximum_seconds=max(r['seconds'] for r in rr),
                       timing_scope=sorted({r['timing_scope'] for r in rr}))
        return out

    studies=TABLE_STUDIES+['independent_r2','independent_r3']
    accuracy={study:{m:summarize([r for r in rows if r['study']==study and r['method']==m]) for m in ('S','NC','C')} for study in studies}
    timings={study:{m:summarize([r for r in runtime_rows if r['study']==study and r['method']==m],True) for m in ('NC','C')} for study in studies if study!='linear1d'}
    for study,group in timings.items():
        paired=[]
        a={(r['geometry'],r['seed']):r for r in runtime_rows if r['study']==study and r['method']=='NC'}
        b={(r['geometry'],r['seed']):r for r in runtime_rows if r['study']==study and r['method']=='C'}
        for key in a.keys() & b.keys():
            paired.append(dict(geometry=key[0],seed=key[1],seconds_nc=a[key]['seconds'],seconds_c=b[key]['seconds'],ratio_nc_over_c=a[key]['seconds']/b[key]['seconds']))
        group['paired']=paired
        group['median_paired_ratio_nc_over_c']=statistics.median(r['ratio_nc_over_c'] for r in paired) if paired else None

    variance_summary=read(HERE/'variance/summary.json')
    variance_manifest=read(HERE/'variance/manifest.json')
    variance_files=[]
    if variance_manifest:
        for group in ('sources','inputs','outputs'):
            for rel,expected in variance_manifest[group].items():
                path=HERE/'variance'/rel
                assert sha(path)==expected,('variance artifact hash',rel)
        variance_files=list((HERE/'variance/results').glob('s*_seed*_*.json'))
        assert len(variance_files)==90
    read(HERE/'independent/accuracy/historical-comparison.json',False)
    read(HERE/'independent/historical-comparison.json',False)
    read(HERE/'first-order/linear1d-current-protocol-audit.json')
    plan=read(HERE/'campaign-plan.json')
    supplementary,supplementary_hashes=inspect_supplementary(HERE/'supplementary-2d',require_complete=False)
    inputs.update({'supplementary-2d/'+name:digest for name,digest in supplementary_hashes.items()})
    inputs['audit_supplementary.py']=sha(HERE/'audit_supplementary.py')
    if not supplementary['complete']:
        missing.append(f"supplementary-2d: {supplementary['found_count']}/116 complete (separate cohort)")
    assert len(rows)==len({(r['study'],r['geometry'],r['seed'],r['method']) for r in rows})
    assert len(runtime_rows)==len({(r['study'],r['geometry'],r['seed'],r['method']) for r in runtime_rows})
    if missing and not args.partial:
        raise SystemExit('Incomplete campaign: '+str(len(missing))+' required records missing. Use --partial for an explicit live snapshot.\n'+'\n'.join(missing[:20]))
    if not missing:
        assert len(rows)==258,len(rows)
        assert len(runtime_rows)==54,len(runtime_rows)
    suffix='.partial' if args.partial else ''
    result=dict(complete=not missing,expected_accuracy_records=258,found_accuracy_records=len(rows),
                expected_new_guarded_accuracy_records=152,
                found_new_guarded_accuracy_records=sum(r['provenance']=='fresh guarded accuracy rerun' for r in rows),
                expected_controlled_benchmark_records=54,found_controlled_benchmark_records=len(runtime_rows),
                fresh_variance_records=len(variance_files),historical_baseline_files=77,
                expected_supplementary_records=116,found_supplementary_records=supplementary['found_count'],
                missing=missing,accuracy=accuracy,controlled_runtime=timings,variance=variance_summary,
                supplementary=supplementary,hardware=hardware,directional_launcher_history=launcher_history,
                pairing_checks=pairing,directional_constructor_provenance=directional_provenance,
                current_source_hashes_verified={str(Path(path).relative_to(WORKSPACE)):digest for path,digest in source_checks.items()},
                runtime_policy=plan['timing_policy'],prespecified_benchmark_subset=plan['benchmark_subset'],
                scope_note='Historical S and all linear1d endpoints remain historical; accuracy runtimes are excluded from runtime comparisons. Controlled NC/C benchmark samples are separate from full accuracy cohorts. Directional controlled runs alone use the separately hashed restart-safe-v1 derived wrapper; their states, coefficients, events, fits, iterations, joins, completions, errors, guard counts and stop status must exactly match the frozen accuracy records. Independent accuracy/timing uses the original frozen guarded source. Variance and supplemental 2D controls are separate cohorts.',
                inputs_sha256=inputs,summary_source_sha256=sha(Path(__file__)))
    (HERE/f'master-summary{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')

    table=[]
    for study in TABLE_STUDIES:
        group=accuracy[study]
        ns=[v['n'] for v in group.values() if v]
        n=str(ns[0]) if len(ns)==3 and len(set(ns))==1 else '/'.join(map(str,ns))+' partial'
        table.append(' & '.join([LABELS[study],n]+[sci(group[m]['median_error']) if group[m] else '--' for m in ('S','NC','C')]
                    +[str(group[m]['successes_1e6']) if group[m] else '--' for m in ('S','NC','C')]
                    +[numeric(group[m]['median_fits']) if group[m] else '--' for m in ('S','NC','C')])+r'\\')
    (HERE/f'table2-rows{suffix}.tex').write_text('\n'.join(table)+'\n')
    table=[]
    for order in (2,3):
        for method in ('S','NC','C'):
            r=accuracy[f'independent_r{order}'][method]
            if r:
                table.append(' & '.join([str(order),method,sci(r['median_error'],2),f"{r['successes_1e6']}/{r['n']}",f"{r['successes_1e10']}/{r['n']}",numeric(r['median_fits'])])+r'\\')
    (HERE/f'table1-rows{suffix}.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Study & $n$ & NC (s) & C (s) & NC successes & C successes\\',r'\midrule']
    for study,group in timings.items():
        nc,c=group['NC'],group['C']
        if not nc or not c: continue
        n=str(nc['n']) if nc['n']==c['n'] else f"{nc['n']}/{c['n']} partial"
        table.append(' & '.join([LABELS[study],n,f"{nc['median_seconds']:.3f}",f"{c['median_seconds']:.3f}",str(nc['successes_1e6']),str(c['successes_1e6'])])+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/f'table-runtime{suffix}.tex').write_text('\n'.join(table)+'\n')
    for label,rr in [('accuracy-endpoints',rows),('benchmark-endpoints',runtime_rows)]:
        if not rr:continue
        fields=[k for k in rr[0] if k!='guard_counts']
        with (HERE/f'{label}{suffix}.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in rr)
    print(json.dumps(dict(complete=not missing,accuracy=len(rows),fresh_guarded=sum(r['provenance']=='fresh guarded accuracy rerun' for r in rows),controlled_benchmarks=len(runtime_rows),variance=len(variance_files),missing=len(missing)),indent=2))


if __name__=='__main__':
    main()
