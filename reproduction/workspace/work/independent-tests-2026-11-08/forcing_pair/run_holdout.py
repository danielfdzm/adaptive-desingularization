#!/usr/bin/env python3
"""Frozen fresh-parameter confirmation in the selected narrow-source regime."""
import json,hashlib
import numpy as np
import run as r
import run_raw

def main():
    spec=json.loads((r.HERE/'holdout_protocol.json').read_text());out=r.HERE/'holdout';out.mkdir(exist_ok=True);rows=[];refs=[]
    for case in spec['cases']:
        a,A,lam,seed=case['a'],case['A'],case['lam'],case['seed']
        coarse=r.reference(a,A,lam,22,1e-10);ref=r.reference(a,A,lam,24,1e-10)
        x=np.linspace(-24,24,19201);w=r.weights(x);u,du=ref.sol(x);f=r.forcing(x,a,A);norm=w@(u*u+du*du);jstar=r.energy(u,du,f,w,lam)
        xx=np.linspace(-22,22,8801);ww=r.weights(xx);u1,d1=coarse.sol(xx);u2,d2=ref.sol(xx);delta=np.sqrt(ww@((u1-u2)**2+(d1-d2)**2)/norm)
        np.savez(out/f'fine-reference-a{a}-A{A}-lambda{lam}.npz',x=x,u=u,du=du,f=f)
        refs.append(dict(**case,reference_energy=jstar,reference_h1_norm=float(np.sqrt(norm)),domain22_vs24_relative_h1=float(delta),nonlinear_force_ratio=float(np.sqrt(w@(lam*(np.sinh(u)-u))**2)/np.sqrt(w@(f*f))),max_bvp_rms_residual=float(np.max(ref.rms_residuals))))
        (out/'references.json').write_text(json.dumps(refs,indent=2))
        for method in spec['methods']:
            if method.startswith('S_'):
                row=run_raw.one(a,A,lam,seed,method,'holdout');row['stored_coefficient_norm']=row['ordinary_weight_norm'];row['stable_coefficient_norm']=float(np.linalg.norm(row['stable_coefficients']))
            else:
                row=r.run_one(a,A,lam,seed,method);B,Bx,*_=r.basis(x,row['beta'],row['h']**2);c=np.array(row['coefficients']);v=c@B;dv=c@Bx
                row['validation_energy']=r.energy(v,dv,f,w,lam);row['energy_gap']=row['validation_energy']-jstar;row['relative_h1']=float(np.sqrt(w@((v-u)**2+(dv-du)**2)/norm));row['training_validation_discrepancy']=row['validation_energy']-row['training_energy'];row['stable_coefficient_norm']=float(np.linalg.norm(c));row['stored_coefficient_norm']=float(np.linalg.norm(c));row['ordinary_weight_norm']=float(np.sqrt((c[0]**2+(c[1]/row['h'])**2)/2)) if row['h'] else None
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
            print(seed,method,'a,A,lam',a,A,lam,'relH1',row['relative_h1'],'gap',row['energy_gap'],'h',row['h'],'calls',row['calls'],flush=True)
    (out/'provenance.json').write_text(json.dumps({str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [r.HERE/'holdout_protocol.json',r.HERE/'run.py',r.HERE/'run_raw.py',r.HERE/'run_holdout.py']},indent=2))
if __name__=='__main__':main()
