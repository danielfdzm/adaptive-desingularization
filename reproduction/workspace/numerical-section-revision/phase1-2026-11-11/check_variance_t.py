import sys, numpy as np
sys.path.insert(0, '.')
import variance_t as vt
beta = 0.13
worst = 0.0
for t in (0.0, 1e-8, 1e-4, 0.0025*(1-1e-9), 0.0025*(1+1e-9), 0.01, 0.3):
    e, d, ex, dx, eb, db, exb, dxb, et, dt, ext, dxt = vt.basis_t(beta, t)
    dl = 1e-7 if t > 0 else 1e-9
    tp = t + dl
    tm = max(t - dl, 0.0)
    P = vt.basis_t(beta, tp); M = vt.basis_t(beta, tm)
    fd = [(P[i] - M[i]) / (tp - tm) for i in (0, 1, 2, 3)]
    errs = [float(np.max(np.abs(a - b))) for a, b in zip((et, dt, ext, dxt), fd)]
    worst = max(worst, max(errs))
    print(f"t={t:.3e} max|analytic-FD| et,dt,ext,dxt = " + ", ".join(f"{x:.1e}" for x in errs))
print("worst", worst)
