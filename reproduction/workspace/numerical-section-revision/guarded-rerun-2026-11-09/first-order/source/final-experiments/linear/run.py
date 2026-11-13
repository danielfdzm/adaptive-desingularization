#!/usr/bin/env python3
"""Reproducible whole-space modified-Helmholtz dimension and robustness studies."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse, copy, csv, hashlib, json, math, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
import torch
import mpmath as mp
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'experiments/core'))
from coefficient_fit import solve_unconstrained_qp
from kernels import block_gram, basis_primitives, kappa, high_precision_error, _mp_primitives, _mp_inner
from base_solver import PairOperations, Geometry, WorkLimit, StructureChange, plain, arrays, offsets, projected_gradient_max, classify_termination
SIGMA=.45
METHODS=('standard','stable','adaptive')

def unit(rng,d):
    z=rng.normal(size=d); return z/np.linalg.norm(z)

def make_problem(task):
    d,geom=task['dimension'],task['geometry']
    rng=np.random.default_rng(14321+geom)
    family=task['family']
    if family=='offmodel':
        centers=np.stack([.6*unit(rng,d) for _ in range(3)])
        return {'kind':'forcing','d':d,'centers':centers,'widths':np.array([.36,.54,.7]),
                'amplitudes':np.array([1.,-.7,.8]),'N':task['capacity']}
    q=d if family=='complexity' else 4
    internal=d if family=='complexity' else 5
    rot_rng=np.random.default_rng(30214+geom+100*d)
    rotation=np.linalg.qr(rot_rng.normal(size=(d,d)))[0][:,:internal]
    centers=np.stack([.70*unit(rng,internal) for _ in range(q+2)])@rotation.T
    directions=np.stack([unit(rng,internal) for _ in range(q)])@rotation.T
    blocks=[]; coefficients=[]
    if family=='control':
        for j in range(6):
            blocks.append(dict(kind='single',c=centers[j]))
            coefficients.append((-1)**j*(.8+.1*j))
    else:
        for j in range(q):
            blocks.append(dict(kind='pair',c=centers[j],e=directions[j],h=0.,completed=True))
            coefficients.extend(((-1)**j*(.7+.15*(j%4)), (-1)**(j//2)*SIGMA*(.8+.1*(j%3))))
        for j in range(2):
            blocks.append(dict(kind='single',c=centers[q+j])); coefficients.append([.65,-.5][j])
    coefficients=np.array(coefficients)
    norm=math.sqrt(coefficients@block_gram(blocks).numpy()@coefficients)
    coefficients/=norm # Linear normalization does not change relative problem geometry.
    N=task.get('capacity',len(coefficients))
    return {'kind':'manufactured','d':d,'blocks':blocks,'coefficients':coefficients,'N':N,
            'centers':centers,'directions':directions,'geometric_rank':int(np.linalg.matrix_rank(np.vstack((centers[1:]-centers[0],directions))))}

def make_starts(problem,task):
    d=problem['d']; rng=np.random.default_rng(71200+task['seed']+100*task['geometry'])
    mode=task.get('initialization','near'); N=problem['N']
    if mode=='independent' or problem['kind']=='forcing':
        return np.stack([rng.uniform(.1,.95)*unit(rng,d) for _ in range(N)])
    amplitude=.065 if mode=='near' else .25
    centers=[]
    for block in problem['blocks']:
        c=block['c']+amplitude*unit(rng,d)
        if block['kind']=='single': centers.append(c)
        else:
            tangent=unit(rng,d); tangent-=np.dot(tangent,block['e'])*block['e']; tangent/=np.linalg.norm(tangent)
            angle=.12 if mode=='near' else .45
            direction=np.cos(angle)*block['e']+np.sin(angle)*tangent
            h=rng.uniform(.02,.05)
            centers.extend((c+h*direction,c-h*direction))
    while len(centers)<N: centers.append(rng.uniform(.1,.95)*unit(rng,d))
    return np.asarray(centers[:N])

def forcing_load(blocks,problem):
    ca,va,aa,T=basis_primitives(blocks)
    cb=torch.tensor(problem['centers']); widths=torch.tensor(problem['widths'])
    s2=SIGMA**2+widths**2
    z=ca[:,None,:]-cb[None,:,:]
    L=(2*math.pi*SIGMA**2*widths**2/s2)**(problem['d']/2)*torch.exp(-torch.sum(z*z,dim=-1)/(2*s2))
    primitive=(aa[:,None]+torch.sum(va[:,None,:]*z,dim=-1)/s2)*L
    return T.T@primitive@torch.tensor(problem['amplitudes'])

def forcing_norm(problem,eps=1e-12):
    c,a,w=problem['centers'],problem['amplitudes'],problem['widths']; d=problem['d']
    dist=np.sum((c[:,None]-c[None,:])**2,axis=-1)
    def value(t):
        S=w[:,None]**2+w[None,:]**2+2*t
        H=(2*math.pi*w[:,None]**2*w[None,:]**2/S)**(d/2)*np.exp(-dist/(2*S))
        return np.exp(-t)*(a@H@a)
    integral,error=quad(value,0,np.inf,epsabs=1e-30,epsrel=eps,limit=200)
    return float(integral),float(error)

def forcing_precise_error(blocks,coef,problem,target2):
    with mp.workdps(70):
        A=_mp_primitives(blocks,coef); norm2=_mp_inner(A,A,SIGMA); load=mp.mpf(0)
        for c,v,a in A:
            for cc,amp,width in zip(problem['centers'],problem['amplitudes'],problem['widths']):
                s2=mp.mpf(SIGMA)**2+mp.mpf(float(width))**2
                z=[x-mp.mpf(float(y)) for x,y in zip(c,cc)]
                L=(2*mp.pi*mp.mpf(SIGMA)**2*mp.mpf(float(width))**2/s2)**(mp.mpf(problem['d'])/2)*mp.exp(-sum(x*x for x in z)/(2*s2))
                load+=mp.mpf(float(amp))*L*(a+sum(x*y for x,y in zip(v,z))/s2)
        e2=norm2-2*load+mp.mpf(target2)
        if e2 < 0:
            raise ArithmeticError('negative independently evaluated forcing error squared')
        return {'relative':float(mp.sqrt(e2/target2)),'norm':float(mp.sqrt(e2)),'target_norm':math.sqrt(target2)}

class Solver(PairOperations):
    def __init__(self,task,max_fits):
        self.task=task; self.d=task['dimension']; self.seed=task['seed']; self.method=task['method']; self.max_fits=max_fits
        self.problem=make_problem(task); self.tb=self.problem.get('blocks'); self.tc=self.problem.get('coefficients')
        self.rho0,self.rho1=kappa(self.d,0),2*kappa(self.d,1)
        if self.problem['kind']=='manufactured':
            self.target2=float(self.tc@block_gram(self.tb).numpy()@self.tc); self.R=float(self.rhos(self.tb)@abs(self.tc)); self.C=None
            self.reference_error=0.
        else:
            self.target2,self.reference_error=forcing_norm(self.problem)
            c,w,a=self.problem['centers'],self.problem['widths'],self.problem['amplitudes']
            S=w[:,None]**2+w[None,:]**2
            F=(2*math.pi*w[:,None]**2*w[None,:]**2/S)**(self.d/2)*np.exp(-np.sum((c[:,None]-c[None,:])**2,axis=-1)/(2*S))
            # Uses forcing only, no unknown solution coefficient cost.
            self.C=None; self.R=None
        self.starts=make_starts(self.problem,task)
        self.blocks=[dict(id=str(i),kind='single',c=c.copy()) for i,c in enumerate(self.starts)]
        self.coef=np.zeros(len(self.starts)); self.warm=None; self.cache=None
        self.fits=self.iterations=self.scans=self.allowance_index=0
        self.events=[]; self.trace=[]; self.flags={}; self.optimizer_returns=[]
        self.completions=self.rejections=self.abnormal=0; self.max_cond=0.; self.best_rel=math.inf
        self.t0=time.perf_counter(); self.error_evaluation_seconds=0.; self.threshold_times={}; self.last_precise=None

    def rhos(self,blocks):
        return np.array([r for b in blocks for r in ([self.rho0] if b['kind']=='single' else [self.rho0,self.rho1])])

    def matrices(self,blocks,order=8):
        M=block_gram(blocks,order=order)
        b=(block_gram(blocks,self.tb,order=order)@torch.tensor(self.tc) if self.tb is not None else forcing_load(blocks,self.problem))
        return M,b

    def evaluate(self,p,geometry):
        if self.cache is not None and np.array_equal(p,self.cache['p']): return self.cache['objective'],self.cache['gradient']
        if self.fits>=self.max_fits: raise WorkLimit
        self.fits+=1
        pt=torch.tensor(p,requires_grad=True); blocks=geometry.unpack(pt)
        Mt,bt=self.matrices(blocks); M,b=Mt.detach().numpy(),bt.detach().numpy(); M=(M+M.T)/2
        scale=np.sqrt(np.maximum(np.diag(M),1e-300)); Mn=M/scale[:,None]/scale[None,:]
        cond=float(np.linalg.cond(Mn)); self.max_cond=max(self.max_cond,cond)
        g,info=solve_unconstrained_qp(M/self.target2,b/self.target2)
        if not np.all(np.isfinite(g)): raise FloatingPointError('nonfinite_coefficients')
        gt=torch.tensor(g); proxy=(.5*gt@Mt@gt-bt@gt)/self.target2
        grad=torch.autograd.grad(proxy,pt)[0].detach().numpy(); obj=float(proxy.detach())
        if not math.isfinite(obj) or not np.all(np.isfinite(grad)): raise FloatingPointError('nonfinite_objective_or_gradient')
        residual=M@g-b
        info.update(kkt_scaled_residual=float(max(abs(residual)/scale))/math.sqrt(self.target2),
                    budget_violation=None,cost=float(self.rhos(blocks)@np.abs(g)),multiplier=0.0)
        self.cache=dict(p=np.array(p),blocks=arrays(plain(blocks)),coef=g.copy(),objective=obj,gradient=grad,cond=cond,fit_info=info,projected_gradient_max_abs=projected_gradient_max(p,grad,geometry.bounds))
        return obj,grad

    def precise_error(self,blocks=None,coef=None):
        blocks=self.blocks if blocks is None else blocks; coef=self.coef if coef is None else coef
        start=time.perf_counter()
        e=high_precision_error(blocks,coef,self.tb,self.tc) if self.tb is not None else forcing_precise_error(blocks,coef,self.problem,self.target2)
        self.error_evaluation_seconds+=time.perf_counter()-start
        return e

    def energy(self,blocks,coef,precise=False):
        if precise: return .5*self.precise_error(blocks,coef)['norm']**2
        M,b=self.matrices(blocks); return float(.5*coef@M.numpy()@coef-coef@b.numpy()+.5*self.target2)

    def record(self,precise=False):
        rawgap=self.energy(self.blocks,self.coef); estimate=math.sqrt(max(0.,2*rawgap/self.target2))
        # Roundoff-prone small errors and all final errors are checked independently.
        if precise or estimate<2e-6 or self.fits==1:
            e=self.precise_error(); relative=e['relative']; self.last_precise=relative
        else: relative=estimate
        wall=time.perf_counter()-self.t0
        for threshold in (1e-3,1e-4,1e-5,1e-6,1e-7):
            if relative<=threshold and str(threshold) not in self.threshold_times: self.threshold_times[str(threshold)]=wall
        self.best_rel=min(self.best_rel,relative)
        M=block_gram(self.blocks).numpy(); scale=np.sqrt(np.maximum(np.diag(M),1e-300))
        row=dict(fits=self.fits,iterations=self.iterations,wall=wall,relative_error=relative,energy_gap=.5*relative**2*self.target2,completions=self.completions,coefficient_norm=float(np.linalg.norm(self.coef)),cost=float(self.rhos(self.blocks)@abs(self.coef)),gram_condition=float(np.linalg.cond(M/scale[:,None]/scale[None,:])),error_independently_checked=bool(precise or estimate<2e-6 or self.fits==1))
        self.trace.append(row); return row

    def transfer(self,i):
        block=self.blocks[i]; ix=offsets(self.blocks); a,beta=self.coef[ix[i]:ix[i+1]]
        trial=arrays(self.blocks); trial[i].update(completed=True,h=0.)
        bound=block['h']**2*(abs(a)*kappa(self.d,2)/2+abs(beta)*kappa(self.d,3)/6)
        norm=math.sqrt(max(0.,float(self.coef@block_gram(self.blocks).numpy()@self.coef)))
        before=self.energy(self.blocks,self.coef,True); after=self.energy(trial,self.coef,True)
        epsilon=1e-3*self.target2*2.**(-self.allowance_index-1)
        accepted=bound<=.01*norm and after-before<=epsilon
        event=dict(kind='completion' if accepted else 'transfer_rejected',fits=self.fits,block_id=block['id'],h=block['h'],state_error_bound=bound,state_tolerance=.01*norm,gate=after-before,eps_j=epsilon,before_blocks=plain(self.blocks),before_coefficients=self.coef.tolist(),trial_blocks=plain(trial),trial_coefficients=self.coef.tolist())
        self.events.append(event)
        if not accepted: self.rejections+=1; return False
        self.blocks=trial; self.allowance_index+=1; self.completions+=1; self.warm=self.coef.copy(); self.fit_current(); self.record()
        return True

    def scan(self):
        self.scans+=1; changed=self.regroup()
        if self.method=='stable': return changed
        for i,b in enumerate(self.blocks):
            if b['kind']!='pair' or b['completed']: continue
            ratio=self.diagnostics(i)
            flag=2*b['h']<=.5*SIGMA and ratio<=.1
            self.flags[b['id']]=self.flags.get(b['id'],0)+1 if flag else 0
            self.events.append(dict(kind='scan',fits=self.fits,block_id=b['id'],flag=flag,persistence=self.flags[b['id']],h=b['h'],gram_smallest_singular_value=ratio))
            if self.flags[b['id']]>=2 and self.transfer(i): return True
        return changed

    def run(self):
        stop='work_limit'
        try:
            self.fit_current(); self.record()
            if self.method!='standard': self.regroup(); self.fit_current(); self.record()
            stalls=0
            while self.fits<self.max_fits:
                geometry=Geometry(self.blocks,self.d); self.cache=None; self.warm=self.coef.copy()
                def callback(x):
                    self.evaluate(x,geometry); self.accept_cached(); self.iterations+=1
                    if self.iterations%20==0:
                        self.record()
                        if self.method!='standard' and self.scan(): raise StructureChange
                    if self.method=='standard' and self.cache['cond']>1e16: raise FloatingPointError('ordinary_gram_condition_exceeds_1e16')
                try:
                    result=minimize(lambda p:self.evaluate(p,geometry),geometry.x,jac=True,method='L-BFGS-B',bounds=geometry.bounds,callback=callback,options=dict(ftol=1e-15,gtol=1e-10,maxls=30,maxiter=3000))
                except StructureChange: stalls=0; continue
                self.evaluate(result.x,geometry); self.accept_cached(); self.record()
                termination=classify_termination(result,geometry.x,self.cache['gradient'],geometry.bounds); self.optimizer_returns.append(dict(fits=self.fits,**termination))
                if result.status==2 or termination['classification']=='numerical_stagnation': self.abnormal+=1
                if self.method=='standard': stop=termination['classification']; break
                if self.scan(): stalls=0; continue
                stalls+=1
                if stalls>=2: stop=termination['classification']; break
        except WorkLimit: stop='work_limit'
        except (FloatingPointError,np.linalg.LinAlgError) as exc: stop=str(exc); self.abnormal+=1
        final=self.record(True)
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return plain(dict(**self.task,initial_centers=self.starts,problem=self.problem,blocks=self.blocks,coefficients=self.coef,C=self.C,R_target=self.R,target_norm=math.sqrt(self.target2),reference_quad_error=self.reference_error,stop=stop,fits=self.fits,iterations=self.iterations,wall=time.perf_counter()-self.t0,error_evaluation_seconds=self.error_evaluation_seconds,peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),peak_rss_scope='worker lifetime high-water mark',final_relative_error=final['relative_error'],best_relative_error=self.best_rel,energy_gap=final['energy_gap'],cost=final['cost'],completions=self.completions,abnormal_returns=self.abnormal,projected_gradient_max_abs=getattr(self,'accepted_projected_gradient',None),coefficient_fit=getattr(self,'accepted_fit_info',None),threshold_times=self.threshold_times,optimizer_returns=self.optimizer_returns,trace=self.trace,events=self.events))

def task_id(t):
    return f"{t['family']}_d{t['dimension']}_g{t['geometry']}_s{t['seed']}_{t.get('initialization','near')}_N{t.get('capacity','default')}_unconstrained_{t['method']}"

def task_list(suite):
    tasks=[]
    def add(family,ds,geoms,seeds,**kw):
        for d in ds:
            for g in geoms:
                for seed in seeds:
                    for method in METHODS:
                        tasks.append(dict(family=family,dimension=d,geometry=g,seed=seed,method=method,**kw))
    if suite=='pilot':
        add('ambient',[5,10],[0],[901]); add('offmodel',[10],[0],[901],capacity=8)
        add('complexity',[10],[0],[901]); return tasks
    if suite in ('all','ambient'): add('ambient',[5,10,20],[0,1],range(10,15))
    if suite in ('all','supplementary','paper'):
        add('complexity',[10,20],[0],range(10,13))
    if suite in ('all','supplementary'):
        add('initialization',[10],[0],range(10,13),initialization='moderate')
        add('initialization',[10],[0],range(10,13),initialization='independent')
        for capacity in (6,8,12): add('capacity',[10],[0],[10],capacity=capacity)
        for capacity in (8,16): add('offmodel',[10],[0],range(10,13),capacity=capacity,initialization='independent')
        add('control',[10],[0],range(10,13))
    return tasks

def run_one(task,max_fits): return Solver(task,max_fits).run()

def summarize(out,tasks):
    rows=[]
    for task in tasks:
        p=out/'runs'/f'{task_id(task)}.json'
        if p.exists(): rows.append(json.loads(p.read_text()))
    groups={}
    for row in rows:
        key=(row['family'],row['dimension'],row['method'],row.get('initialization','near'),row.get('capacity','default'),None)
        groups.setdefault(key,[]).append(row)
    summary=[]
    for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        errors=np.array([r['final_relative_error'] for r in rs]); walls=np.array([r['wall'] for r in rs])
        summary.append(dict(family=key[0],dimension=key[1],method=key[2],initialization=key[3],capacity=key[4],budget_factor=key[5],runs=len(rs),success_1e6=int(sum(errors<=1e-6)),error_median=float(np.median(errors)),error_min=float(min(errors)),error_max=float(max(errors)),error_q25=float(np.quantile(errors,.25)),error_q75=float(np.quantile(errors,.75)),wall_median=float(np.median(walls)),fits_median=float(np.median([r['fits'] for r in rs])),completions_median=float(np.median([r['completions'] for r in rs])),work_limits=sum(r['stop']=='work_limit' for r in rs),abnormal_runs=sum(r['abnormal_returns']>0 for r in rs),gradient_max=max(r['projected_gradient_max_abs'] or 0 for r in rs),kkt_max=max(r['coefficient_fit']['kkt_scaled_residual'] for r in rs if r['coefficient_fit'])))
    (out/'summary.json').write_text(json.dumps(dict(expected_runs=len(tasks),completed_runs=len(rows),groups=summary),indent=2)+'\n')
    if summary:
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--suite',choices=['pilot','ambient','supplementary','paper','all'],default='all'); ap.add_argument('--jobs',type=int,default=2); ap.add_argument('--max-fits',type=int,default=1200); ap.add_argument('--output',type=Path,default=HERE/'unconstrained-results'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--summarize-only',action='store_true'); args=ap.parse_args()
    tasks=task_list(args.suite); out=args.output; out.mkdir(parents=True,exist_ok=True); (out/'runs').mkdir(exist_ok=True)
    if args.summarize_only: summarize(out,tasks); return
    sources=[HERE/x for x in ('run.py','kernels.py','base_solver.py')]+[HERE.parents[1]/'experiments/core/coefficient_fit.py']
    provenance=dict(coefficient_constraint=None,suite=args.suite,max_fits=args.max_fits,jobs=args.jobs,threads_per_worker=1,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,platform=platform.platform(),command=sys.argv,source_hashes={str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},tasks=tasks)
    pp=out/'provenance.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        if not args.resume: raise SystemExit('Output has provenance; use --resume or a fresh output directory.')
        for key in ('suite','max_fits','source_hashes','tasks'):
            if old[key]!=provenance[key]: raise SystemExit('Cannot resume with changed '+key)
    pp.write_text(json.dumps(provenance,indent=2)+'\n')
    pending=[t for t in tasks if not (out/'runs'/f'{task_id(t)}.json').exists()]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,t,args.max_fits):t for t in pending}
        for future in as_completed(futures):
            task=futures[future]; result=future.result(); (out/'runs'/f'{task_id(task)}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(task_id(task),f"error={result['final_relative_error']:.3e} wall={result['wall']:.1f}s fits={result['fits']} completed={result['completions']} stop={result['stop']}",flush=True)
            summarize(out,tasks)
    provenance['elapsed_seconds']=time.perf_counter()-start; pp.write_text(json.dumps(provenance,indent=2)+'\n'); summarize(out,tasks)

if __name__=='__main__': main()
