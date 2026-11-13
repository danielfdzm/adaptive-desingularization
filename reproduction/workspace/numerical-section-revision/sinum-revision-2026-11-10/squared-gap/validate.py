#!/usr/bin/env python3
"""Independent value/gradient checks, including the t=0 boundary."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import mpmath as mp
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source'))
import kernels
import run
import squared_gap as tchart


def main():
    rng = np.random.default_rng(98331)
    stats = dict(value_error=0., gram_error=0., relative_gram_error=0.,
                 interior_gradient_error=0., zero_point_derivative_error=0.,
                 zero_gram_derivative_error=0., boundary_path_error=0.)
    for d in (5, 10, 20):
        blocks = []
        for i, h in enumerate((0., 1e-10, 1e-5, .03, .27)):
            e = rng.normal(size=d); e /= np.linalg.norm(e)
            blocks.append(dict(kind='pair', id=str(i), c=rng.normal(size=d)*.2,
                               e=e, h=h, t=h*h, completed=False))
        blocks.append(dict(kind='single', id='5', c=rng.normal(size=d)*.2))
        X = torch.tensor(rng.normal(size=(35, d))*.4)
        a = tchart.column_values(blocks, X)
        b = kernels.column_values(blocks, X)
        stats['value_error'] = max(stats['value_error'], float((a-b).abs().max()))
        M = tchart.block_gram(blocks)
        Mref = kernels.block_gram(blocks, order=24)
        stats['gram_error'] = max(stats['gram_error'], float((M-Mref).abs().max()))
        stats['relative_gram_error'] = max(stats['relative_gram_error'], float((M-Mref).norm()/Mref.norm()))
        # Envelope proxy derivatives at positive t equal (d/dh)/(2h).
        for h in (.03, .27):
            hh = torch.tensor(h, requires_grad=True)
            tt = torch.tensor(h*h, requires_grad=True)
            old = [dict(blocks[3], h=hh)]
            new = [dict(blocks[3], h=h, t=tt)]
            f = kernels.block_gram(old, order=24).sum()+kernels.column_values(old, X).square().sum()
            g = tchart.block_gram(new).sum()+tchart.column_values(new, X).square().sum()
            dh = torch.autograd.grad(f, hh)[0]/(2*h)
            dt = torch.autograd.grad(g, tt)[0]
            stats['interior_gradient_error'] = max(stats['interior_gradient_error'],
                float(abs(dt-dh)/max(1., abs(dh))))
        t = torch.tensor(0., requires_grad=True)
        b0 = dict(blocks[0], t=t)
        vals = tchart.column_values([b0], X)
        y = X-torch.tensor(b0['c'])
        s = y@torch.tensor(b0['e'])/.45**2
        G = torch.exp(-(y*y).sum(-1)/(2*.45**2))
        expected = torch.stack((G*(s*s/2-1/(2*.45**2)),
                                G*s*(s*s/6-1/(2*.45**2))), dim=-1)
        for col in (0, 1):
            dt = torch.autograd.grad(vals[:, col].sum(), t, retain_graph=True)[0]
            stats['zero_point_derivative_error'] = max(stats['zero_point_derivative_error'],
                float(abs(dt-expected[:, col].sum())))
        # Independent 90-digit kernel derivatives of the original shifted
        # Gaussian formula: E derivative uses F''(0)/2, D uses F'''(0)/6.
        right = dict(blocks[4], h=.03, t=.03**2)
        cross = tchart.block_gram([b0], [right])
        with mp.workdps(90):
            ca = [mp.mpf(float(v)) for v in b0['c']]
            ea = [mp.mpf(float(v)) for v in b0['e']]
            cb = [mp.mpf(float(v)) for v in right['c']]
            eb = [mp.mpf(float(v)) for v in right['e']]
            hb, sigma = mp.mpf(right['h']), mp.mpf(.45)
            def F(u, q):
                answer = mp.mpf(0)
                for sign in (-1, 1):
                    z = [x+u*v-y-sign*hb*w for x,v,y,w in zip(ca,ea,cb,eb)]
                    rr = sum(v*v for v in z)
                    K = mp.pi**(mp.mpf(d)/2)*sigma**d*mp.exp(-rr/(4*sigma*sigma))*(1+d/(2*sigma*sigma)-rr/(4*sigma**4))
                    answer += K*(1 if q == 0 else sign/hb)/2
                return answer
            for p in (0, 1):
                for q in (0, 1):
                    ref = float(mp.diff(lambda u:F(u,q), 0, p+2)/math.factorial(p+2))
                    got = float(torch.autograd.grad(cross[p,q], t, retain_graph=True)[0])
                    stats['zero_gram_derivative_error'] = max(stats['zero_gram_derivative_error'], abs(got-ref)/max(1.,abs(ref)))
        # A finite sequence approaches the boundary value and derivative.
        for tt in (1e-4, 1e-8, 1e-12, 0.):
            z = torch.tensor(tt, requires_grad=True)
            bb = dict(b0, t=z, h=math.sqrt(tt))
            yy = tchart.column_values([bb], X)
            dy = torch.autograd.grad(yy.sum(), z)[0]
            assert torch.isfinite(dy)
            if tt <= 1e-12:
                stats['boundary_path_error'] = max(stats['boundary_path_error'],
                    float(abs(dy-expected.sum())))
    # Verify the original NC and T produce identical initial grouping and
    # coefficients before either begins geometric optimization.
    nc = run.Solver(5,10,'stable',12,1500,0,8128,1.)
    nc.fit_current(); nc.regroup()
    before_blocks, before_coef = run.plain(nc.blocks), nc.coef.copy()
    tchart.install(run)
    continuous = run.Solver(5,10,'squared_gap',12,1500,0,8128,1.)
    continuous.fit_current(); continuous.regroup()
    assert run.plain(continuous.blocks) == before_blocks
    assert np.array_equal(continuous.coef, before_coef)
    stats['initial_pairing_and_coefficients_identical'] = True
    geo = tchart.Geometry(continuous.blocks, 5)
    t_indices = [i for i,b in enumerate(geo.bounds) if b == (0.,(.6*.45)**2)]
    assert len(t_indices) == 4
    geo.x[t_indices] = 0.
    continuous.cache = None
    objective, grad = continuous.evaluate(geo.x, geo)
    assert np.all(np.isfinite(grad)) and np.isfinite(objective)
    continuous.accept_cached(); row = continuous.record()
    assert row['zero_squared_gaps'] == 4
    assert all(not b.get('completed',False) for b in continuous.blocks)
    # Differentiate the actual PB objective along one-sided feasible paths,
    # using the original h-based kernels and frozen coefficients throughout.
    # At a fitted coefficient vector its derivative is the envelope gradient.
    coef = continuous.coef.copy()
    def original_energy(p):
        bb = run.arrays(run.plain(geo.unpack(torch.tensor(p))))
        for item in bb:
            item.pop('t',None)
        M = kernels.block_gram(bb,order=24).numpy()
        b = kernels.block_gram(bb,continuous.tb,order=24).numpy()@continuous.tc
        V = kernels.column_values(bb,continuous.Xt).numpy()@coef
        return (.5*coef@M@coef-b@coef+run.weighted_bregman(V,continuous.U,continuous.W))/continuous.target2
    base = original_energy(geo.x)
    energy_errors=[]
    for i in t_indices:
        step=1e-5
        p1=geo.x.copy();p1[i]+=step
        p2=geo.x.copy();p2[i]+=2*step
        fd=(-3*base+4*original_energy(p1)-original_energy(p2))/(2*step)
        energy_errors.append(abs(fd-grad[i]))
    stats['zero_PB_energy_derivative_error']=max(energy_errors)
    assert max(energy_errors)<3e-7
    # Boundary variables remain free and a positive t step reopens each pair.
    reopened = geo.x.copy(); reopened[t_indices] = 1e-6
    bb = geo.unpack(torch.tensor(reopened))
    assert all(b.get('h',1.) > 0 for b in bb)
    stats['boundary_optimizer_gradient_finite'] = True
    stats['boundary_is_reversible'] = True
    stats['boundary_projected_gradient'] = run.projected_gradient_max(geo.x,grad,geo.bounds)
    assert stats['value_error'] < 3e-14
    assert stats['relative_gram_error'] < 2e-13
    assert stats['interior_gradient_error'] < 3e-11
    assert stats['zero_point_derivative_error'] < 1e-11
    assert stats['zero_gram_derivative_error'] < 2e-12
    assert stats['boundary_path_error'] < 1e-7
    (HERE/'validation.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
