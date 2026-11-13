"""Unconstrained quadratic coefficient fits, without penalties or rank truncation."""
from __future__ import annotations
import numpy as np

def solve_unconstrained_qp(M, b):
    """Solve M g=b after diagonal scaling; never regularize or clip coefficients.

    Failure of a singular or numerically inconsistent system is reported to the
    optimizer rather than replaced by a constrained or truncated solution.
    """
    M=np.asarray(M,dtype=float); b=np.asarray(b,dtype=float)
    M=(M+M.T)/2
    diagonal=np.diag(M)
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(b)) or np.any(diagonal<=0):
        raise FloatingPointError("invalid_coefficient_system")
    scale=np.sqrt(diagonal)
    normalized=M/scale[:,None]/scale[None,:]
    rhs=b/scale
    y=np.linalg.solve(normalized,rhs)
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("nonfinite_coefficients")
    residual=normalized@y-rhs
    denominator=np.linalg.norm(normalized,ord=np.inf)*np.linalg.norm(y,ord=np.inf)+np.linalg.norm(rhs,ord=np.inf)
    backward=float(np.linalg.norm(residual,ord=np.inf)/max(denominator,np.finfo(float).tiny))
    if backward>1e-10:
        raise FloatingPointError("coefficient_solve_residual")
    return y/scale, dict(constrained=False,budget=None,penalty=0.0,rank_truncation=False,
                        solver="diagonally_scaled_direct",backward_error=backward,
                        active=int(np.count_nonzero(y)),lam=0.0,iters=0)
