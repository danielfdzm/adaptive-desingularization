#!/usr/bin/env python3
"""Declared energy-gate sensitivity; all original cases/seeds are retained."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,importlib.util,json,math,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_2d',HERE/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class StrictGate(m.IndependentSolver):
    def __init__(self,*args,energy_factor=1e-9,**kwargs):
        self.energy_factor=energy_factor
        super().__init__(*args,**kwargs)

    def transfer(self,i):
        # Identical to the inherited transfer except the declared energy factor.
        b=self.blocks[i];ix=m.base.offsets(self.blocks)
        a,beta=self.coef[ix[i]:ix[i+1]]
        trial=m.base.arrays(self.blocks);trial[i].update(completed=True,h=0.)
        error_bound=b['h']**2*(abs(a)*m.kappa(self.d,2)/2+abs(beta)*m.kappa(self.d,3)/6)
        norm=math.sqrt(max(float(self.coef@m.block_gram(self.blocks).numpy()@self.coef),0.))
        before_energy,_=self.energy(self.blocks,self.coef,precise=True)
        trial_energy,_=self.energy(trial,self.coef,precise=True)
        gate=trial_energy-before_energy
        eps=self.energy_factor*self.target2*2.**(-self.allowance_index-1)
        event=dict(kind='completion' if error_bound<=.01*norm and gate<=eps else 'transfer_rejected',
            fits=self.fits,block_id=b['id'],h=b['h'],direction=b['e'].copy(),
            state_error_bound=error_bound,state_tolerance=.01*norm,gate=gate,eps_j=eps,
            before_blocks=m.base.plain(self.blocks),before_coefficients=self.coef.tolist(),
            trial_blocks=m.base.plain(trial),trial_coefficients=self.coef.tolist())
        if event['kind']=='transfer_rejected':
            self.rejections+=1;self.events.append(event);return False
        self.blocks=trial;self.allowance_index+=1;self.completions+=1
        self.warm=self.coef.copy();self.events.append(event);self.fit_current()
        after_energy,_=self.energy(self.blocks,self.coef,precise=True)
        event.update(after_blocks=m.base.plain(self.blocks),after_coefficients=self.coef.tolist(),
                     dJ_after_refit=after_energy-before_energy)
        self.record();return True

def run_one(task):
    case,seed,factor=task
    r=StrictGate(case,seed,'adaptive',max_fits=1000,order=64,perturbation=.01,capacity=4,
                 initialization='lattice',energy_factor=factor).run()
    r['energy_factor']=factor
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    tasks=[(c,s,f) for f in [1e-6,1e-9] for c in ['dipole','tilted'] for s in range(101,106)]
    sources=[HERE/'sensitivity.py',HERE/'run.py',*[m.BASE/n for n in ['run.py','fit.py','kernels.py','quadrature.py','budgetqp.py']]]
    provenance=dict(command=sys.argv,tasks=tasks,only_changed_parameter='energy_factor; original1e-3, sensitivity1e-6 and1e-9',
       source_hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    runs=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for fut in as_completed(futures):
            c,s,f=futures[fut]
            try:r=fut.result()
            except Exception as exc:r=dict(case=c,seed=s,energy_factor=f,exception=repr(exc))
            (args.output/f'{c}_{s}_gate{f:g}.json').write_text(json.dumps(r,indent=2)+'\n');runs.append(r)
            print(c,s,f,{k:r.get(k) for k in ['final_relative_error','fits','completions','transfer_rejections','exception']},flush=True)
    summaries=[]
    for f in [1e-6,1e-9]:
        for c in ['dipole','tilted']:
            rr=[r for r in runs if r['case']==c and r['energy_factor']==f]
            good=[r for r in rr if 'final_relative_error' in r]
            summaries.append(dict(case=c,energy_factor=f,runs=len(rr),exceptions=len(rr)-len(good),
                median_error=float(np.median([r['final_relative_error'] for r in good])),
                median_fits=float(np.median([r['fits'] for r in good])),
                completions=sum(r['completions'] for r in good),rejections=sum(r['transfer_rejections'] for r in good)))
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
