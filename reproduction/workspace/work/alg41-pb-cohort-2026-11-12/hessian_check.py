#!/usr/bin/env python3
"""Finite-difference Hessian of the reduced energy on T_A at the final iterate of each Algorithm 4.1 run.

Checks the hypothesis of Theorem 4.1(ii) where the runs end, and compares the observed rate q with
1 - alpha * lambda_min (alpha = 1), the contraction of the projected-gradient step on the slowest direction.
The reduced energy extends analytically to t < 0, so central differences are used in every direction.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alg41 as A          # noqa: E402
from analyze import rate   # noqa: E402


def grad_vector(prob, z, Y):
    ev = prob.evaluate(z)
    g = ev["grad"]
    full = np.concatenate([g["beta"].ravel(), g["t"], g["e"].ravel(), g["c"].ravel()])
    return Y.T @ full, ev


def basis(z, d, strictly_active):
    m, s = A.M_PAIRS, A.S_SINGLE
    nb, nt, ne, nc = m * d, m, m * d, s * d
    n = nb + nt + ne + nc
    cols = []
    for i in range(nb):
        v = np.zeros(n); v[i] = 1; cols.append(v)
    for j in range(nt):
        if j in strictly_active:
            continue
        v = np.zeros(n); v[nb + j] = 1; cols.append(v)
    for j in range(m):
        U = null_space(z["e"][j].reshape(1, d))
        for k in range(d - 1):
            v = np.zeros(n); v[nb + nt + j * d:nb + nt + (j + 1) * d] = U[:, k]; cols.append(v)
    for i in range(nc):
        v = np.zeros(n); v[nb + nt + ne + i] = 1; cols.append(v)
    return np.array(cols).T


def unflat(x, d):
    m, s = A.M_PAIRS, A.S_SINGLE
    nb, nt, ne = m * d, m, m * d
    return dict(beta=x[:nb].reshape(m, d).copy(), t=x[nb:nb + nt].copy(),
                e=x[nb + nt:nb + nt + ne].reshape(m, d).copy(), c=x[nb + nt + ne:].reshape(s, d).copy())


def check(rec, eps=1e-5):
    d, g, s = rec["d"], rec["geometry"], rec["seed"]
    prob = A.Problem(d, g, s)
    fs = rec["final_state"]
    z = dict(beta=np.array(fs["beta"]), t=np.array(fs["t"]), e=np.array(fs["e"]), c=np.array(fs["c"]))
    prob.warm = np.array(fs["coef"])
    ev = prob.evaluate(z)
    gt = ev["grad"]["t"]
    strictly = {j for j in range(A.M_PAIRS) if z["t"][j] == 0.0 and gt[j] > 1e-6}
    Y = basis(z, d, strictly)
    x0 = np.concatenate([z["beta"].ravel(), z["t"], z["e"].ravel(), z["c"].ravel()])
    H = np.zeros((Y.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        gp, _ = grad_vector(prob, unflat(x0 + eps * Y[:, i], d), Y)
        gm, _ = grad_vector(prob, unflat(x0 - eps * Y[:, i], d), Y)
        H[:, i] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)
    lam = np.linalg.eigvalsh(H)
    r = rate(rec["trace"])
    return dict(d=d, geometry=g, seed=s, dim=int(Y.shape[1]), strictly_active=sorted(strictly),
                lambda_min=float(lam[0]), lambda_max=float(lam[-1]),
                observed_q=(r["q"] if r else None), predicted_q=float(max(abs(1 - lam[0]), abs(1 - lam[-1]))),
                final_t=[float(t) for t in z["t"]], final_residual=rec["final_residual"])


def main():
    out = []
    for p in sorted((HERE / "results" / "pg_t").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("stop") in ("not_run", "exception"):
            continue
        res = check(rec)
        out.append(res)
        print(json.dumps(res), flush=True)
    (HERE / "hessian_check.json").write_text(json.dumps(out, indent=1) + "\n")
    write_table(out)


def sci(x):
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"


def write_table(out):
    rows = []
    for d in (5, 10, 20):
        rr = [r for r in out if r["d"] == d]
        if not rr:
            continue
        lam = np.array([r["lambda_min"] for r in rr])
        qo = np.array([r["observed_q"] for r in rr if r["observed_q"] is not None])
        qp = np.array([r["predicted_q"] for r in rr])
        rows.append(f"{d} & {len(rr)} & {rr[0]['dim']} & {sci(lam.min())} & {sci(float(np.median(lam)))} & "
                    f"{float(np.median(qo)):.4f} & {float(np.median(qp)):.4f}\\\\")
    (HERE / "table_hessian_rows.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
