#!/usr/bin/env python3
"""Validate and summarize every saved matched variance outcome, no optimization."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import mpmath as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math_support as exact

HERE=Path(__file__).resolve().parent


def main():
    files=sorted((HERE/'results').glob('s*_seed*_*.json'))
    runs=[json.loads(p.read_text()) for p in files]
    assert len(runs)==90,len(runs)
    summary=[]
    validations=[]
    for variance in (.99,.995,.999):
        with mp.workdps(80):
            reference=float(exact.reference_summary(str(variance))['relative_error'])
        for method in ('S','NC','C'):
            rr=[r for r in runs if r['variance']==variance and r['method']==method]
            assert sorted(r['seed'] for r in rr)==list(range(10))
            summary.append(dict(variance=variance,method=method,runs=10,reference_error=reference,
                                within_one_percent=sum(r['final']['relative_error']<=1.01*reference for r in rr),
                                median_error=statistics.median(r['final']['relative_error'] for r in rr),
                                max_error=max(r['final']['relative_error'] for r in rr),
                                median_iterations=statistics.median(r['iterations'] for r in rr),
                                median_objective_evaluations=statistics.median(r['objective_evaluations'] for r in rr),
                                median_optimizer_seconds=statistics.median(r['optimizer_wall_seconds'] for r in rr),
                                total_optimizer_seconds=sum(r['optimizer_wall_seconds'] for r in rr),
                                median_stored_coefficient_norm=statistics.median(r['final']['stored_coefficient_norm'] for r in rr),
                                median_ordinary_weight_norm=statistics.median(r['final']['ordinary_weight_norm'] for r in rr) if method!='C' else None,
                                completions=sum(r['completion_count'] for r in rr),
                                half_gap_floor_hits=sum(r['hit_half_gap_floor'] for r in rr),
                                final_abnormal_returns=sum('ABNORMAL' in r['stop'] for r in rr)))
        for seed in range(10):
            pair={r['method']:r for r in runs if r['variance']==variance and r['seed']==seed}
            assert pair['S']['initial_parameters']==pair['NC']['initial_parameters']==pair['C']['initial_parameters']
            # Both stable methods follow exactly the same optimization before the first completion.
            cc=pair['C']
            accepted=[e for e in cc['events'] if e['accepted']]
            if accepted:
                step=accepted[0]['iteration']
                before=[p for p in cc['trace'] if p['phase']=='stable' and p['iteration']<=step]
                nt=pair['NC']['trace']
                assert len(nt)>=len(before)
                for a,b in zip(before,nt):
                    for key in ('a','c','beta','h','iteration','objective_evaluations','relative_error'):
                        assert a[key]==b[key],(variance,seed,key,a[key],b[key])
            with mp.workdps(80):
                for event in accepted:
                    p=event['before'];ss=mp.mpf(str(variance))
                    bef=exact.pair_evaluation(ss,*[mp.mpf(p[k]) for k in ('beta','h','a','c')])
                    aft=exact.pair_evaluation(ss,mp.mpf(p['beta']),mp.mpf(0),mp.mpf(p['a']),mp.mpf(p['c']))
                    change=float(aft['energy_gap']-bef['energy_gap'])
                    tolerance=1e-13*cc['target_norm_squared']
                    assert change<=event['energy_increase_bound']+tolerance
                    assert event['state_error_measured']<=event['state_error_bound']+tolerance
                    validations.append(dict(variance=variance,seed=seed,
                                            continuous_energy_change_80digit=change,
                                            continuous_energy_upper_bound=event['energy_increase_bound'],
                                            state_change_measured=event['state_error_measured'],
                                            state_bound=event['state_error_bound']))
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'validation.json').write_text(json.dumps(dict(matched_initializations=True,
        stable_trajectories_identical_until_completion=True,accepted_event_checks=validations),indent=2)+'\n')
    with (HERE/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    endpoint_fields=['variance','seed','method','relative_error','optimizer_seconds','iterations','objective_evaluations','stored_coefficient_norm','half_gap','completions','stop']
    with (HERE/'endpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=endpoint_fields);writer.writeheader()
        for r in runs:
            writer.writerow(dict(variance=r['variance'],seed=r['seed'],method=r['method'],
                relative_error=r['final']['relative_error'],optimizer_seconds=r['optimizer_wall_seconds'],
                iterations=r['iterations'],objective_evaluations=r['objective_evaluations'],
                stored_coefficient_norm=r['final']['stored_coefficient_norm'],half_gap=r['final']['h'],
                completions=r['completion_count'],stop=r['stop']))
    lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
           r'$s$ & Method & Runs & Median $E$ & Iter. & Time (s) & Norm\\',r'\midrule']
    for r in summary:
        lines.append(f"{r['variance']:g} & {r['method']} & {r['within_one_percent']}/10 & "
                     f"${r['median_error']:.3g}$ & {r['median_iterations']:g} & "
                     f"{r['median_optimizer_seconds']:.3f} & ${r['median_stored_coefficient_norm']:.3g}$"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (HERE/'table_variance_matched.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({
        'font.family':'serif',
        'font.serif':['Latin Modern Roman','DejaVu Serif'],
        'font.size':9,
        'mathtext.fontset':'cm',
        'axes.titlesize':10,
        'axes.titleweight':'normal',
        'axes.labelsize':9,
        'legend.fontsize':8,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
    })
    colors={'S':'#0000FF','NC':'#188038','C':'#FF0000'}
    styles={'S':'-','NC':'--','C':':'}
    fig,axes=plt.subplots(1,2,figsize=(7,3.05))
    for run in (r for r in runs if r['variance']==.99):
        trace=run['trace'];m=run['method']
        xx=[max(p['elapsed_seconds'],1e-5) for p in trace]
        for ax,key in zip(axes,('relative_error','stored_coefficient_norm')):
            ax.loglog(xx,[p[key] for p in trace],color=colors[m],ls=styles[m],lw=.85,alpha=.65)
        for event in run['events']:
            if event['accepted']:
                match=next(p for p in trace if p['phase']=='completed')
                axes[0].plot(match['elapsed_seconds'],match['relative_error'],'*',color=colors[m],ms=5)
                axes[1].plot(match['elapsed_seconds'],match['stored_coefficient_norm'],'*',color=colors[m],ms=5)
    reference=next(r['reference_error'] for r in summary if r['variance']==.99)
    axes[0].axhline(reference,color='#555555',ls=':',lw=1)
    axes[0].set(ylabel=r'Relative $H^1$ error',title='(a) Approximation error')
    axes[1].set(ylabel='Stored coefficient norm',title='(b) Coefficient growth')
    for ax in axes:
        ax.set_xlabel('Optimizer time (s)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    fig.legend(handles=[Line2D([],[],color=colors[m],ls=styles[m],label=m) for m in ('S','NC','C')]
               +[Line2D([],[],color=colors['C'],marker='*',ls='none',label='Completion')],
               loc='upper center',ncol=4,bbox_to_anchor=(.52,1.0),frameon=False)
    fig.tight_layout(rect=(0,0,1,.87))
    for suffix in ('pdf','png'):
        fig.savefig(HERE/f'figure_variance_matched.{suffix}',bbox_inches='tight',dpi=300)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    outputs=['summary.json','summary.csv','endpoints.csv','validation.json','table_variance_matched.tex','figure_variance_matched.pdf','figure_variance_matched.png']
    (HERE/'manifest.json').write_text(json.dumps(dict(
        sources={p.name:sha(p) for p in (HERE/'run_matched.py',HERE/'summarize.py',HERE/'math_support.py')},
        inputs={str(p.relative_to(HERE)):sha(p) for p in files+[HERE/'results/protocol.json']},
        outputs={name:sha(HERE/name) for name in outputs}),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
