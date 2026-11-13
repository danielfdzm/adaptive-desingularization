#!/usr/bin/env python3
"""Independent-start, off-dictionary 2D Poisson--Boltzmann comparison.

Training uses only explicitly evaluated f and fixed positive quadrature, with
unconstrained coefficients. Exact u is used by the independent error monitor.
The existing PB implementation supplies fitting, grouping and completion.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, copy, csv, hashlib, importlib.util, json, math, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'pb'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('pb_base_run',BASE/'run.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from kernels import block_gram, column_values, kappa, _mp_primitives
from quadrature import weighted_bregman

def rule(order=64, halfwidth=4.):
    z,w=np.polynomial.legendre.leggauss(order)
    z,w=z*halfwidth,w*halfwidth
    X,Y=np.meshgrid(z,z,indexing='ij')
    return np.column_stack((X.ravel(),Y.ravel())),np.outer(w,w).ravel()

def target(case, perturbation=.01):
    if case=='dipole':
        return [dict(c=[.18,-.11],sigma=.45,a=0.,v=[-.55*math.cos(.6),-.55*math.sin(.6)]),
                dict(c=[-.42,.31],sigma=.8,a=perturbation,v=[0.,0.])]
    if case=='tilted':
        return [dict(c=[-.23,.16],sigma=.45,a=.25,v=[-.65*math.cos(1.7),-.65*math.sin(1.7)]),
                dict(c=[.45,-.38],sigma=.72,a=-perturbation,v=[0.,0.])]
    raise ValueError(case)

def field(X, terms):
    u=np.zeros(len(X)); grad=np.zeros_like(X); lap=np.zeros(len(X))
    for p in terms:
        y=X-np.array(p['c']); s2=p['sigma']**2; v=np.array(p['v'])
        r=(y*y).sum(1); vy=y@v; g=np.exp(-r/(2*s2)); L=r/s2**2-2/s2
        u += (p['a']-vy/s2)*g
        grad += (-(p['a']*y+v)/s2 + vy[:,None]*y/s2**2)*g[:,None]
        lap += (p['a']*L + vy*(2/s2**2-L/s2))*g
    return u,grad,lap

def inner(A,B):
    total=mp.mpf('0')
    for ca,va,aa,sa in A:
        for cb,vb,ab,sb in B:
            S=sa*sa+sb*sb; z=[x-y for x,y in zip(ca,cb)]
            r=sum(x*x for x in z); alpha=1/(2*S); beta=1/S**2
            pref=(2*mp.pi*sa*sa*sb*sb/S)*mp.exp(-alpha*r)
            pol=1+2/S-beta*r
            f0=pref*pol; f1=pref*(-alpha*pol-beta); f2=pref*(alpha**2*pol+2*alpha*beta)
            za=sum(x*y for x,y in zip(z,va)); zb=sum(x*y for x,y in zip(z,vb))
            vv=sum(x*y for x,y in zip(va,vb))
            total += aa*ab*f0+2*f1*(aa*zb-ab*za)-4*f2*za*zb-2*f1*vv
    return total

def target_primitives(terms):
    return [([mp.mpf(float(x)) for x in t['c']], [mp.mpf(float(x)) for x in t['v']],
             mp.mpf(float(t['a'])),mp.mpf(float(t['sigma']))) for t in terms]

def precise_error(blocks,coef,terms):
    with mp.workdps(70):
        A=[(*p,mp.mpf(.45)) for p in _mp_primitives(blocks,coef)]
        T=target_primitives(terms)
        norm2=inner(T,T); error2=inner(A,A)-2*inner(A,T)+norm2
        if error2 < -mp.mpf('1e-45'): raise ArithmeticError('negative high precision error')
        return float(mp.sqrt(max(error2,mp.mpf(0))/norm2)),float(mp.sqrt(max(error2,mp.mpf(0)))),float(mp.sqrt(norm2))

class IndependentSolver(base.Solver):
    def __init__(self,case,seed,method,max_fits=800,order=64,perturbation=.01,capacity=2,initialization='uniform'):
        self.t0=time.perf_counter(); self.d=2; self.seed=seed; self.method=method; self.max_fits=max_fits
        self.case=case; self.terms=target(case,perturbation); self.geometry_id=0; self.quadrature_seed=0
        self.reaction_strength=1.; self.tb=[]; self.tc=[]
        self.X,self.Q=rule(order); self.W=self.Q.copy(); self.Xt=torch.as_tensor(self.X)
        exact,_,lap=field(self.X,self.terms)
        self.force=exact-lap+np.sinh(exact)-exact
        self.target2=float(self.Q@(self.force**2)) # force-only scale, not target norm
        self.C=None; self.R=None
        self.rho0,self.rho1=kappa(2,0),2*kappa(2,1)
        self.U=np.zeros(len(self.X)) # F(v), no target-dependent sampled Bregman objective
        self.chi_nl=None; self.target_sampled_peak=None
        self.qmeta=dict(rule='tensor_gauss_legendre',nodes_per_axis=order,halfwidth=4.,nodes=len(self.Q),
                        positive_weights=True,target_geometry_used=False)
        rng=np.random.default_rng(seed)
        self.initialization=initialization
        if initialization=='uniform':
            self.starts=rng.uniform(-.9,.9,size=(capacity,2))
        elif initialization=='lattice':
            if capacity==4:
                self.starts=np.array([[-.65,-.65],[-.65,.65],[.65,-.65],[.65,.65]])
            elif capacity==6:
                self.starts=np.array([[x,y] for x in [-.7,0.,.7] for y in [-.55,.55]])
            else:
                raise ValueError('lattice starts support capacity four or six')
            self.starts+=rng.uniform(-.12,.12,size=self.starts.shape)
        else:raise ValueError(initialization)
        self.blocks=[dict(id=str(i),kind='single',c=c.copy()) for i,c in enumerate(self.starts)]
        self.coef=np.zeros(capacity)
        self.fits=self.iterations=self.scans=self.allowance_index=0
        self.events=[];self.trace=[];self.flags={};self.optimizer_returns=[]
        self.cache=self.warm=None;self.completions=self.rejections=self.abnormal=0
        self.max_cond=0.;self.best_rel=math.inf
        self.setup_wall=time.perf_counter()-self.t0
        self.quadrature_nodes_total=self.nonlinear_derivative_calls=self.nonlinear_value_calls=0

    def matrices(self,blocks):
        M=block_gram(blocks); Phi=column_values(blocks,self.Xt)
        b=Phi.T@torch.as_tensor(self.Q*self.force)
        return M,b,Phi

    def energy(self,blocks,coef,precise=False):
        M,b,Phi=self.matrices(blocks)
        nonlin=weighted_bregman(Phi.detach().numpy()@coef,self.U,self.W)
        return float(.5*coef@M.detach().numpy()@coef-coef@b.detach().numpy()+nonlin),nonlin

    def record(self):
        rel,err,norm=precise_error(self.blocks,self.coef,self.terms)
        J,nonlin=self.energy(self.blocks,self.coef)
        self.best_rel=min(self.best_rel,rel)
        M=block_gram(self.blocks).numpy(); diag=np.sqrt(np.diag(M))
        raw=[]; gaps=[];ix=base.offsets(self.blocks)
        for i,b in enumerate(self.blocks):
            coef=self.coef[ix[i]:ix[i+1]]
            if b['kind']=='single' or b.get('completed',False):raw.extend(coef)
            else:
                a,c=coef;h=b['h'];raw.extend((a/2+c/(2*h),a/2-c/(2*h)));gaps.append(h)
        row=dict(fits=self.fits,iterations=self.iterations,wall=time.perf_counter()-self.t0,
                 relative_error=rel,error_norm=err,target_norm=norm,energy_gap=J,nonlinear_gap=nonlin,
                 coefficient_norm=float(np.linalg.norm(self.coef)),ordinary_finite_part_norm=float(np.linalg.norm(raw)),
                 min_positive_halfgap=min(gaps) if gaps else None,cost=float(self.rhos(self.blocks)@abs(self.coef)),
                 completions=self.completions,stored_gram_condition=float(np.linalg.cond(M/diag[:,None]/diag[None,:])))
        self.trace.append(row)
        return row

    def run(self):
        result=super().run()
        result.update(case=self.case,exact_target=self.terms,budget_rule=None,coefficient_constraint='none',
                      objective_normalization='divide by squared L2 norm of forcing',
                      initialization=self.initialization,
                      target_norm=self.trace[-1]['target_norm'],force_norm=math.sqrt(self.target2),
                      training_consumes_only_evaluated_forcing=True,capacity=len(self.starts),
                      forcing_is_manufactured=True,
                      energy_gap_field_means='absolute sampled objective, not an excess above its unknown minimum',
                      target_outside_fixed_width_completed_dictionary=True)
        X,W=rule(96); U,_,_=field(X,self.terms);_,_,lap=field(X,self.terms);f=np.sinh(U)-lap
        V=column_values(self.blocks,torch.as_tensor(X)).numpy()@self.coef
        M=block_gram(self.blocks).numpy()
        Jref=float(.5*self.coef@M@self.coef-W@(f*V)+weighted_bregman(V,np.zeros_like(V),W))
        result['energy_refinement_abs_difference']=abs(Jref-result['energy_gap'])
        result['energy_validation_96']=Jref
        X5,W5=rule(120,5.);U5,_,lap5=field(X5,self.terms);f5=np.sinh(U5)-lap5
        V5=column_values(self.blocks,torch.as_tensor(X5)).numpy()@self.coef
        J5=float(.5*self.coef@M@self.coef-W5@(f5*V5)+weighted_bregman(V5,np.zeros_like(V5),W5))
        result['energy_validation_halfwidth5']=J5
        result['energy_domain_refinement_abs_difference']=abs(J5-Jref)
        result['target_norm']=precise_error(self.blocks,self.coef,self.terms)[2]
        return result

def run_one(task):
    case,seed,method,max_fits,order,perturbation,capacity,initialization=task
    return IndependentSolver(case,seed,method,max_fits,order,perturbation,capacity,initialization).run()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['dipole','tilted'])
    p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);p.add_argument('--methods',nargs='+',default=['standard','stable','adaptive'])
    p.add_argument('--max-fits',type=int,default=800);p.add_argument('--order',type=int,default=64)
    p.add_argument('--perturbation',type=float,default=.01);p.add_argument('--capacity',type=int,default=2)
    p.add_argument('--initialization',choices=['uniform','lattice'],default='uniform')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tasks=[(c,s,m,args.max_fits,args.order,args.perturbation,args.capacity,args.initialization) for c in args.cases for s in args.seeds for m in args.methods]
    sources=[Path(__file__).resolve(),*[BASE/n for n in ['run.py','fit.py','kernels.py','quadrature.py','budgetqp.py']]]
    provenance=dict(command=sys.argv,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                    source_hashes={str(x.relative_to(HERE.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                    protocol=vars(args).copy(),retention='Every requested run including failed outcomes; all pilot directories retained.')
    provenance['protocol']['output']=str(args.output)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:r=future.result()
            except Exception as exc:r=dict(case=task[0],seed=task[1],method=task[2],exception=repr(exc))
            (args.output/f'{task[0]}_{task[1]}_{task[2]}.json').write_text(json.dumps(r,indent=2)+'\n')
            results.append(r)
            print(task[:3],{k:r.get(k) for k in ['final_relative_error','fits','completions','wall','stop','exception']},flush=True)
    keys=['case','seed','method','final_relative_error','fits','completions','wall','stop','cost','C','coefficient_norm','energy_refinement_abs_difference','exception']
    with (args.output/'runs.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in results)
    summaries=[]
    for case in args.cases:
        for method in args.methods:
            runs=[r for r in results if r['case']==case and r['method']==method]
            good=[r for r in runs if 'final_relative_error' in r]
            summaries.append(dict(case=case,method=method,runs=len(runs),exceptions=len(runs)-len(good),
                **{key+'_median':float(np.median([r[key] for r in good])) if good else None for key in ['final_relative_error','fits','wall','completions']},
                errors=[r['final_relative_error'] for r in good]))
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
