#!/usr/bin/env python3
"""Measured trajectories; no extrapolation after a run terminates."""
from __future__ import annotations
import argparse,json,os,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'desingularization-pb-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'desingularization-pb-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'standard':'#b04a35','stable':'#3a78a1','adaptive':'#23835b'}
LABELS={'standard':'Ordinary','stable':'Stable only','adaptive':'Completed'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs=[json.loads(p.read_text()) for p in sorted(args.results.glob('d*_g*_seed*_*.json'))]
    dimensions=sorted({r['dimension'] for r in runs})
    reaction=runs[0]['reaction_strength']
    if any(r['reaction_strength']!=reaction for r in runs):raise ValueError('Plot one reaction strength at a time')
    fig,axes=plt.subplots(len(dimensions),2,figsize=(9.2,2.5*len(dimensions)),squeeze=False,layout='constrained')
    for row,d in enumerate(dimensions):
        for method,color in COLORS.items():
            selected=[r for r in runs if r['dimension']==d and r['method']==method]
            for i,r in enumerate(selected):
                tr=r['trace'];wall=np.array([t['wall'] for t in tr]);err=np.minimum.accumulate([t['relative_error'] for t in tr])
                axes[row,0].semilogy(wall,err,color=color,alpha=.5,lw=1,label=LABELS[method] if i==0 else None)
                if r['stop']=='work_limit' or r['abnormal_returns']:
                    axes[row,0].scatter(wall[-1],err[-1],color=color,marker='x',s=18)
                x=(0 if method=='standard' else 1 if method=='stable' else 2)+.035*(i-(len(selected)-1)/2)
                axes[row,1].scatter(x,r['final_relative_error'],color=color,s=17,alpha=.75)
            if selected:
                med=np.median([r['final_relative_error'] for r in selected]);x=list(COLORS).index(method)
                axes[row,1].plot([x-.22,x+.22],[med,med],color=color,lw=2.5)
        axes[row,0].set_title(f'd = {d}, λ = {reaction:g}: running minimum error',fontsize=10)
        axes[row,0].set_xlabel('Elapsed seconds');axes[row,0].set_ylabel(r'Relative $H^1$ error')
        axes[row,0].grid(alpha=.2)
        axes[row,1].set_title('Final errors: all starts; bar = median',fontsize=10)
        axes[row,1].set_yscale('log');axes[row,1].set_xticks([0,1,2],list(LABELS.values()))
        axes[row,1].grid(axis='y',alpha=.2);axes[row,1].set_ylabel(r'Relative $H^1$ error')
    axes[0,0].legend(frameon=False,fontsize=8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output);fig.savefig(args.output.with_suffix('.png'),dpi=160)

if __name__=='__main__':main()
