"""Independent derivative and unconstrained envelope-gradient checks."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import run
import multi


def main():
    result={}
    x=torch.tensor(np.linspace(-2,2,21),requires_grad=True)
    errors=[]
    for h in [.7,.03,1e-8]:
        P,Q=run.basis(torch.tensor([.13,h]),x,.65,'stable',False)
        ee=[float((torch.autograd.grad(P[:,j].sum(),x,retain_graph=True)[0]-Q[:,j]).abs().max().detach()) for j in range(2)]
        errors.append(dict(half_gap=h,max_errors=ee))
    result['pair_spatial_derivatives']=errors
    sigma,lam,f,_=multi.problem('mixed_width');xx,ww=run.quadrature(384);ff=f(xx);scale=float(ww@(ff*ff))
    xt=torch.tensor(xx);wt=torch.tensor(ww)
    def kap(k):return math.sqrt(math.sqrt(math.pi)*sigma*math.factorial(2*k)/(4**k*math.factorial(k)*sigma**(2*k))*(1+(1+2*k)/(2*sigma*sigma)))
    rho=np.tile([kap(0),2*kap(1)],2);checks=[]
    for done,initial in [([False,False],[-.7,.05,.8,.08]),([True,False],[-.7,.8,.08])]:
      for _ in [None]:
        def evaluate(p):
            pt=torch.tensor(p,requires_grad=True);P,Q=multi.multi_basis(pt,xt,sigma,'stable',done)
            g,info=run.coefficient_fit(P.detach().numpy(),Q.detach().numpy(),ww,ff,lam,rho)
            v=P@torch.tensor(g);dx=Q@torch.tensor(g);vv=v.detach().numpy();dd=dx.detach().numpy()
            value=float(ww@(.5*(vv*vv+dd*dd)+lam*run.rem(vv)-ff*vv))/scale
            vg=torch.tensor(vv+lam*(np.sinh(vv)-vv)-ff)
            gradient=torch.autograd.grad((wt*(vg*v+dx.detach()*dx)).sum(),pt)[0].numpy()/scale
            return value,gradient,info
        p=np.array(initial);_,g,info=evaluate(p);fd=[]
        for k in range(len(p)):
            delta=np.zeros_like(p);delta[k]=1e-5
            fd.append((evaluate(p+delta)[0]-evaluate(p-delta)[0])/2e-5)
        checks.append(dict(completed=done,budget=None,coefficient_constraint='none',cost=info['cost'],kkt=info['kkt'],multiplier=info['multiplier'],gradient_error=float(np.max(np.abs(g-fd)))))
    result['multipair_envelope_gradients']=checks
    result['source_sha256']={n:hashlib.sha256((run.HERE/n).read_bytes()).hexdigest() for n in ['run.py','multi.py','log_gap.py']}
    result['passed']=all(max(r['max_errors'])<2e-12 for r in errors) and all(r['gradient_error']<2e-7 and r['kkt']<1e-8 and r['multiplier']==0 for r in checks)
    (run.HERE/'unconstrained-calculus-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
