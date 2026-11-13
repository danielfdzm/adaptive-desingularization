"""Exact first-order Gaussian pair calculus in the squared halfgap t >= 0.

E_t = exp(-(r+t)/(2 sigma^2)) cosh(s sqrt(t)),
D_t = exp(-(r+t)/(2 sigma^2)) s sinhc(s sqrt(t)), s=(x-c).e/sigma^2.
Both entire functions are evaluated as functions of s^2*t, with analytic
power series near zero. In particular d_t E_0=G*(s^2/2-1/(2 sigma^2))
and d_t D_0=G*s*(s^2/6-1/(2 sigma^2)). No sqrt(t) derivative is used.

For the Gram kernel, parity averaging the exact Gaussian H1 inner product
gives four combinations of the same entire functions. This is algebraically
exact (up to floating point and the near-zero series), and replaces the
historical eight-node derivative-integral evaluation for T only.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.linalg import null_space

SIGMA = .45


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def entire(x):
    """Return cosh(sqrt(x)), sinh(sqrt(x))/sqrt(x), smoothly at x=0."""
    small = x < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    root = torch.sqrt(safe)
    ce, se = torch.cosh(root), torch.sinh(root)/root
    c = 1+x*(1/2+x*(1/24+x*(1/720+x*(1/40320+x/3628800))))
    s = 1+x*(1/6+x*(1/120+x*(1/5040+x*(1/362880+x/39916800))))
    return torch.where(small, c, ce), torch.where(small, s, se)


def _data(blocks):
    centers, directions, gaps, indices = [], [], [], []
    for i, b in enumerate(blocks):
        c = _tensor(b['c'])
        centers.append(c)
        indices.append(2*i)
        if b['kind'] == 'single':
            directions.append(torch.zeros_like(c))
            gaps.append(_tensor(0.))
        else:
            directions.append(_tensor(b['e']))
            gaps.append(_tensor(0. if b.get('completed', False) else b.get('t', b['h']**2)))
            indices.append(2*i+1)
    return torch.stack(centers), torch.stack(directions), torch.stack(gaps), indices


def block_gram(blocks, other_blocks=None, sigma=SIGMA, order=8):
    del order
    ca, ea, ta, ia = _data(blocks)
    cb, eb, tb, ib = _data(blocks if other_blocks is None else other_blocks)
    z = ca[:, None, :] - cb[None, :, :]
    r = (z*z).sum(-1)
    za = (z*ea[:, None, :]).sum(-1)
    zb = (z*eb[None, :, :]).sum(-1)
    v = ea @ eb.T
    ta, tb = ta[:, None], tb[None, :]
    alpha, beta = 1/(4*sigma**2), 1/(4*sigma**4)
    a, b, c = -2*alpha*za, 2*alpha*zb, 2*alpha*v
    A, SA = entire(a*a*ta)
    B, SB = entire(b*b*tb)
    C, SC = entire(c*c*ta*tb)
    pref = (math.pi**(ca.shape[1]/2)*sigma**ca.shape[1])*torch.exp(-alpha*(r+ta+tb))
    S = [[pref*(A*B*C+a*b*c*ta*tb*SA*SB*SC),
          pref*(b*A*SB*C+a*c*ta*SA*B*SC)],
         [pref*(a*SA*B*C+b*c*tb*A*SB*SC),
          pref*(a*b*SA*SB*C+c*A*B*SC)]]
    poly = 1+ca.shape[1]/(2*sigma**2)-beta*(r+ta+tb)
    values = []
    for p in (0, 1):
        row = []
        fa = ta if p == 0 else 1.
        for q in (0, 1):
            fb = tb if q == 0 else 1.
            row.append(poly*S[p][q]-2*beta*za*fa*S[1-p][q]
                       +2*beta*zb*fb*S[p][1-q]+2*beta*v*fa*fb*S[1-p][1-q])
        values.append(torch.stack(row, dim=-1))
    full = torch.stack(values, dim=1).reshape(2*len(blocks), 2*len(cb))
    return full[ia, :][:, ib]


def column_values(blocks, X, sigma=SIGMA):
    X = _tensor(X)
    columns = []
    for b in blocks:
        y = X-_tensor(b['c'])
        r = (y*y).sum(-1)
        if b['kind'] == 'single':
            columns.append(torch.exp(-r/(2*sigma**2)))
            continue
        t = _tensor(0. if b.get('completed', False) else b.get('t', b['h']**2))
        s = (y@_tensor(b['e']))/sigma**2
        C, S = entire(s*s*t)
        pref = torch.exp(-(r+t)/(2*sigma**2))
        columns.extend((pref*C, pref*s*S))
    return torch.stack(columns, dim=-1)


class Geometry:
    """Original center/tangent coordinates, with t replacing the halfgap h."""
    def __init__(self, blocks, d):
        self.blocks, self.d = blocks, d
        self.frames, vals, self.bounds = [], [], []
        for b in blocks:
            vals.extend(b['c'])
            self.bounds.extend([(-2., 2.)]*d)
            if b['kind'] == 'pair':
                self.frames.append(null_space(np.asarray(b['e']).reshape(1, d)))
                vals.extend([0.]*(d-1))
                self.bounds.extend([(-2., 2.)]*(d-1))
                if not b['completed']:
                    vals.append(b.get('t', b['h']**2))
                    self.bounds.append((0., (.6*SIGMA)**2))
            else:
                self.frames.append(None)
        self.x = np.array(vals)

    def unpack(self, p):
        blocks, j = [], 0
        for b, U in zip(self.blocks, self.frames):
            c = p[j:j+self.d]
            j += self.d
            new = dict(id=b['id'], kind=b['kind'], c=c)
            if b['kind'] == 'pair':
                e = _tensor(b['e'])+_tensor(U)@p[j:j+self.d-1]
                j += self.d-1
                e = e/torch.linalg.vector_norm(e)
                new.update(e=e, completed=b['completed'], h=0.)
                if not b['completed']:
                    new['t'] = p[j]
                    # h is retained only for high-precision audits and diagnostics.
                    new['h'] = float(torch.sqrt(p[j].detach()))
                    j += 1
            blocks.append(new)
        return blocks


def install(module):
    """Apply only the T chart changes; preserve all optimization rules."""
    original_gram = module.block_gram
    original_values = module.column_values
    # Ordinary initialization and manufactured target use the frozen kernels.
    module.block_gram = lambda b, other=None, **kw: (
        block_gram(b, other, **kw) if any('t' in x for x in b+(other or []))
        else original_gram(b, other, **kw))
    module.column_values = lambda b, X, **kw: (
        column_values(b, X, **kw) if any('t' in x for x in b)
        else original_values(b, X, **kw))
    module.Geometry = Geometry
    original_record = module.Solver.record

    def record(self):
        boundary = [b for b in self.blocks if b['kind'] == 'pair' and b.get('t') == 0.]
        # At t=0 there are no ordinary atom weights to report. Temporarily mark
        # that diagnostic representation as confluent without freezing its t.
        for b in boundary:
            b['completed'] = True
        try:
            row = original_record(self)
        finally:
            for b in boundary:
                b['completed'] = False
        row['zero_squared_gaps'] = len(boundary)
        row['squared_gaps'] = [b.get('t', b['h']**2) for b in self.blocks if b['kind'] == 'pair']
        return row

    module.Solver.record = record
