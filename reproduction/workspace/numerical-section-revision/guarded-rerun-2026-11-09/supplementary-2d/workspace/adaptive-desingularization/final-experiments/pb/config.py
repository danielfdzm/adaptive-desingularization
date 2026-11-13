"""Fixed-geometry ambient-dimension PB family, with reproducible starts.

The two target geometries are genuinely different five-dimensional geometries;
each is embedded isometrically into d=5,10,20 using a dense orthonormal frame.
Their geometric span remains five-dimensional. The PDE and all learned centers
and directions use every ambient coordinate.
"""
from __future__ import annotations
import numpy as np
from kernels import kappa

SIGMA = .45
N = 10
CENTERS = np.array([[-.55,-.42,-.18,.12,.14], [.47,-.40,.16,-.14,.21],
                   [-.44,.54,.12,.22,-.17], [.57,.43,-.21,-.19,.13],
                   [-.03,.62,.23,.11,.19], [.05,-.56,-.11,.24,-.22]])
ANGLES = np.array([.34,1.26,2.04,2.86])
EXTRAS = np.array([[.28,-.22,.34],[-.31,.26,.21],[.24,.33,-.29],[-.22,-.35,-.27]])
COEFFICIENTS = np.array([1.1,-1.3,-.8,1.5,1.4,.9,-1.2,-1.1,.9,-.7])

def base_geometry(geometry_id=0):
    if geometry_id not in (0,1):
        raise ValueError('geometry_id must be 0 or 1')
    centers = CENTERS.copy()
    directions = np.column_stack((np.cos(ANGLES), np.sin(ANGLES), EXTRAS))
    if geometry_id:
        rng = np.random.default_rng(72831)
        centers += rng.normal(size=centers.shape) * .035
        directions += rng.normal(size=directions.shape) * .05
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    return centers,directions

def embedding(d,geometry_id=0):
    if d < 5:
        raise ValueError('dimension must be at least five')
    rng=np.random.default_rng(43000+100*geometry_id+d)
    q,r=np.linalg.qr(rng.normal(size=(d,5)))
    return q*np.sign(np.diag(r))[None,:]

def target_blocks(d,geometry_id=0):
    c,e=base_geometry(geometry_id)
    q=embedding(d,geometry_id)
    c,e=c@q.T,e@q.T
    blocks=[dict(kind='pair',c=c[i],e=e[i],h=0.,completed=True) for i in range(4)]
    blocks += [dict(kind='single',c=c[i]) for i in range(4,6)]
    return blocks,COEFFICIENTS.copy()

def start_centers(d,seed,geometry_id=0):
    blocks,_=target_blocks(d,geometry_id)
    rng=np.random.default_rng(100000*geometry_id+seed)
    centers=[]
    for b in blocks:
        perturb=rng.normal(size=d)
        perturb *= rng.uniform(.06,.14)/np.linalg.norm(perturb)
        c=b['c']+perturb
        if b['kind']=='pair':
            tangent=rng.normal(size=d)
            tangent -= tangent@b['e']*b['e']
            tangent /= np.linalg.norm(tangent)
            angle=rng.uniform(-.15,.15)
            e=np.cos(angle)*b['e']+np.sin(angle)*tangent
            h=rng.uniform(.02,.05)
            centers.extend((c+h*e,c-h*e))
        else:
            centers.append(c)
    return np.asarray(centers)

def configuration(d,geometry_id=0):
    blocks,coef=target_blocks(d,geometry_id)
    costs=np.array([v for b in blocks for v in
                    ([kappa(d,0),2*kappa(d,1)] if b['kind']=='pair' else [kappa(d,0)])])
    centers=np.array([b['c'] for b in blocks])
    dirs=np.array([b['e'] for b in blocks if b['kind']=='pair'])
    span=np.vstack((centers[1:]-centers[0],dirs))
    return dict(dimension=d,geometry_id=geometry_id,sigma=SIGMA,N=N,pair_count=4,
                singleton_count=2,target_centers=centers.tolist(),target_directions=dirs.tolist(),
                target_coefficients=coef.tolist(),embedding=embedding(d,geometry_id).tolist(),
                geometry_rank=int(np.linalg.matrix_rank(span)),
                geometry_singular_values=np.linalg.svd(span,compute_uv=False).tolist(),
                R_target=float(costs@abs(coef)),C=None,
                C_factor=None,budget_uses_exact_target=False,coefficient_constraint=None,
                center_perturbation='uniform Euclidean radius [.06,.14], random sphere direction',
                direction_perturbation='uniform angle [-.15,.15] in random ambient tangent direction',
                halfgap='uniform [.02,.05]',target_geometric_dimension=5,
                initialization='near-target ordinary centers, no pair labels passed to solver')
