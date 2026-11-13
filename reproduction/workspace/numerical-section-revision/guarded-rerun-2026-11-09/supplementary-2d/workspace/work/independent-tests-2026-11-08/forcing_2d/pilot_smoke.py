"""Prescribed-source 2D PDE screening; every requested outcome is retained.

The source is specified directly, without constructing an exact Gaussian solution.
Training consumes only source values and fixed-box quadrature. An independent
Fourier/Newton PDE solve supplies diagnostics, never optimizer inputs. Existing
pair grouping/completion code is reused without changes. Pilot seeds 200--202;
seeds 900--909 are reserved for a frozen validation protocol.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import LinearOperator, cg
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'adaptive-desingularization/final-experiments/pb-independent-2d'
spec = importlib.util.spec_from_file_location('independent_2d', BASE / 'run.py')
ind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ind)
sys.path.insert(0, str(ROOT / 'revision-2026-11/experiments'))
from pb_variants import closure_geometry, restart_run
from kernels import block_gram, column_values
from quadrature import weighted_bregman

BASE_GEOMETRY = ind.base.Geometry
T_GEOMETRY = closure_geometry(ind.base)
RESTART_RUN = restart_run(ind.base)

# Frozen initial exploration grid. The two-source cases use different directions.
CASES = {
    'dipole_linear': dict(shape='dipole', a=.22, amplitude=18., lam=0., capacity=4),
    'dipole_nonlinear': dict(shape='dipole', a=.22, amplitude=65., lam=10., capacity=4),
    'two_nonlinear': dict(shape='two', a=.22, amplitude=65., lam=10., capacity=6),
    'compact_nonlinear': dict(shape='compact', a=.48, amplitude=180., lam=10., capacity=4),
    'monopole_control': dict(shape='monopole', a=.32, amplitude=40., lam=10., capacity=4),
}


def force(X, cfg):
    def component(c, angle, a, amp, compact=False, monopole=False):
        y = X - np.asarray(c)
        z = y @ np.array([np.cos(angle), np.sin(angle)])
        r2 = np.sum(y*y, axis=1) / a**2
        if compact:
            g = np.zeros(len(X)); mask = r2 < 1
            g[mask] = np.exp(1 - 1/(1-r2[mask]))
        else:
            g = np.exp(-r2/2)
        return amp*g if monopole else amp*z/a**2*g
    a, amp = cfg['a'], cfg['amplitude']
    out = component((.17, -.13), .61, a, amp,
                    cfg['shape']=='compact', cfg['shape']=='monopole')
    if cfg['shape']=='two':
        out = component((-.43, -.29), .61, a, amp)
        out += component((.52, .34), 2.03, 1.2*a, -.8*amp)
    return out


def reference(cfg, n=256, L=8.):
    x = np.linspace(-L, L, n, endpoint=False)
    X,Y = np.meshgrid(x,x,indexing='ij')
    f = force(np.column_stack((X.ravel(),Y.ravel())),cfg).reshape(n,n)
    k = 2*np.pi*fftfreq(n,d=2*L/n)
    KX,KY = np.meshgrid(k,k,indexing='ij'); symbol = 1+KX*KX+KY*KY
    A = lambda v: ifft2(symbol*fft2(v)).real
    u = ifft2(fft2(f)/symbol).real
    lam = cfg['lam']; history=[]
    for it in range(50):
        residual = A(u)+lam*(np.sinh(u)-u)-f
        norm = float(np.max(np.abs(residual))); history.append(norm)
        if norm < 2e-10*max(1.,np.max(np.abs(f))):
            break
        diag = lam*(np.cosh(u)-1)
        op = LinearOperator((n*n,n*n),matvec=lambda z:(A(z.reshape(n,n))+diag*z.reshape(n,n)).ravel())
        pre = LinearOperator((n*n,n*n),matvec=lambda z:ifft2(fft2(z.reshape(n,n))/(symbol+np.mean(diag))).real.ravel())
        step,info = cg(op,-residual.ravel(),M=pre,rtol=1e-11,atol=1e-12,maxiter=300)
        if info: raise RuntimeError(f'CG status {info}')
        step=step.reshape(n,n); t=1.
        for _ in range(30):
            v=u+t*step
            nr=A(v)+lam*(np.sinh(v)-v)-f
            if np.linalg.norm(nr)<np.linalg.norm(residual): break
            t*=.5
        else: raise RuntimeError('reference Newton line search failed')
        u=v
    else: raise RuntimeError('reference Newton limit')
    U=fft2(u); ux=ifft2(1j*KX*U).real; uy=ifft2(1j*KY*U).real
    dx=2*L/n; norm2=float(np.sum(u*u+ux*ux+uy*uy)*dx*dx)
    nonlin=float(weighted_bregman(u.ravel(),np.zeros(n*n),np.full(n*n,lam*dx*dx)))
    return x,u,ux,uy,dict(n=n,L=L,residual_max=history[-1],history=history,
                         peak=float(np.max(np.abs(u))),norm2=norm2,
                         nonlinear_to_quadratic=nonlin/(.5*norm2))


def reference_cached(case, cfg, n=256, L=8.):
    folder=HERE/'references'; folder.mkdir(exist_ok=True)
    path=folder/f'{case}_n{n}_L{L:g}.npz'
    if path.exists():
        r=np.load(path); return r['x'],r['u'],r['ux'],r['uy'],json.loads(str(r['meta']))
    x,u,ux,uy,meta=reference(cfg,n,L)
    np.savez_compressed(path,x=x,u=u,ux=ux,uy=uy,meta=json.dumps(meta))
    return x,u,ux,uy,meta


def values_grad(blocks,g,X,sigma=.45):
    v=np.zeros(len(X)); grad=np.zeros_like(X); j=0; s2=sigma*sigma
    for b in blocks:
        y=X-np.asarray(b['c']); r=np.sum(y*y,axis=1)
        if b['kind']=='single':
            z=np.exp(-r/(2*s2)); v+=g[j]*z; grad-=g[j]*y/s2*z[:,None]; j+=1
        else:
            e=np.asarray(b['e']); h=0. if b['completed'] else b['h']
            s=y@e; t=s*h/s2; q=np.abs(t)
            peak=np.exp(-(r+h*h)/(2*s2)+q)
            phi=np.ones_like(q); mask=q>1e-8
            phi[mask]=-np.expm1(-2*q[mask])/(2*q[mask])
            phi[~mask]=1-q[~mask]+2*q[~mask]**2/3
            E=.5*peak*(1+np.exp(-2*q)); D=peak*s/s2*phi
            v+=g[j]*E+g[j+1]*D
            grad+=(-y/s2*(g[j]*E+g[j+1]*D)[:,None]
                   +(g[j]*h*h*D+g[j+1]*E)[:,None]*e/s2)
            j+=2
    return v,grad


class Solver(ind.IndependentSolver):
    def __init__(self,case,seed,method,max_fits=800,order=72):
        cfg=CASES[case]
        super().__init__('dipole',seed,'stable' if method=='squared' else method,
                         max_fits,order,0.,cfg['capacity'],'lattice')
        self.case=case; self.cfg=cfg; self.variant=method
        self.X,self.Q=ind.rule(order,4.); self.Xt=torch.as_tensor(self.X)
        self.force=force(self.X,cfg); self.target2=float(self.Q@(self.force*self.force))
        self.W=cfg['lam']*self.Q; self.U=np.zeros(len(self.X))
        self.reaction_strength=cfg['lam']; self.tb=[]; self.tc=[]
        self.qmeta.update(halfwidth=4.,target_geometry_used=False)
        x,u,ux,uy,self.refmeta=reference_cached(case,cfg)
        self.VX,self.VW=ind.rule(128,6.)
        self.ref=np.array([RectBivariateSpline(x,x,z,kx=3,ky=3).ev(self.VX[:,0],self.VX[:,1]) for z in (u,ux,uy)])
        self.refnorm2=float(self.VW@np.sum(self.ref*self.ref,axis=0))
        self.chi_nl=self.refmeta['nonlinear_to_quadratic']
        self.target_sampled_peak=self.refmeta['peak']
        self.t0=time.perf_counter()  # Training+monitoring only; reference setup separately recorded.

    def record(self):
        v,dv=values_grad(self.blocks,self.coef,self.VX)
        err2=float(self.VW@((v-self.ref[0])**2+np.sum((dv-self.ref[1:].T)**2,axis=1)))
        rel=math.sqrt(err2/self.refnorm2); self.best_rel=min(self.best_rel,rel)
        J,nl=self.energy(self.blocks,self.coef)
        row=dict(fits=self.fits,iterations=self.iterations,wall=time.perf_counter()-self.t0,
                 relative_error=rel,error_norm=math.sqrt(err2),target_norm=math.sqrt(self.refnorm2),
                 energy_gap=J,nonlinear_gap=nl,coefficient_norm=float(np.linalg.norm(self.coef)),
                 cost=float(self.rhos(self.blocks)@abs(self.coef)),completions=self.completions)
        self.trace.append(row); return row

    def run(self):
        # S gets the same two-return restart convention; all methods retain the same objective.
        result=RESTART_RUN(self)
        result.update(case=self.case,variant=self.variant,source=self.cfg,reference=self.refmeta,
                      objective_normalization='squared training L2 forcing norm',
                      manufactured_solution=False,reference_used_only_for_monitoring=True,
                      energy_gap_field_means='absolute sampled energy',timing_scope='optimization plus diagnostic monitoring')
        result.pop('target_blocks',None); result.pop('target_coefficients',None)
        result['target_norm']=math.sqrt(self.refnorm2)
        checks=[]
        for n,L in [(96,4.),(144,5.)]:
            X,W=ind.rule(n,L); v,_=values_grad(self.blocks,self.coef,X)
            M=block_gram(self.blocks).numpy()
            J=.5*self.coef@M@self.coef-W@(force(X,self.cfg)*v)+weighted_bregman(v,np.zeros_like(v),self.cfg['lam']*W)
            checks.append(dict(order=n,L=L,energy=float(J)))
        result['energy_validation']=checks
        result['energy_refinement_relative']=abs(checks[-1]['energy']-checks[0]['energy'])/max(abs(checks[-1]['energy']),1e-300)
        return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',nargs='+',default=list(CASES))
    ap.add_argument('--seeds',type=int,nargs='+',default=[200,201,202])
    ap.add_argument('--methods',nargs='+',default=['standard','stable','adaptive','squared'])
    ap.add_argument('--max-fits',type=int,default=800); ap.add_argument('--order',type=int,default=72)
    ap.add_argument('--output',default='pilot'); args=ap.parse_args()
    out=HERE/args.output; out.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),cases={c:CASES[c] for c in args.cases},
                  source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
                  retention='all requested outcomes including failures; no holdout tuning')
    old=out/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=manifest:
        raise RuntimeError('changed protocol: use a new output directory')
    old.write_text(json.dumps(manifest,indent=2))
    for case in args.cases:
        for seed in args.seeds:
            for method in args.methods:
                path=out/f'{case}_{seed}_{method}.json'
                if path.exists(): continue
                ind.base.Geometry=T_GEOMETRY if method=='squared' else BASE_GEOMETRY
                try:
                    r=Solver(case,seed,method,args.max_fits,args.order).run()
                except Exception as exc:
                    import traceback
                    r=dict(case=case,seed=seed,variant=method,exception=repr(exc),traceback=traceback.format_exc())
                path.write_text(json.dumps(ind.base.plain(r),indent=1))
                print(case,seed,method,{k:r.get(k) for k in ['final_relative_error','fits','completions','stop','exception']},flush=True)


if __name__=='__main__': main()
