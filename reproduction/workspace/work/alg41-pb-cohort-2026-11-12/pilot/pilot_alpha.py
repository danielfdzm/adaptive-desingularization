import sys, time, math, json
sys.path.insert(0, "..")
import numpy as np
import alg41 as A
d = int(sys.argv[1]); alpha0 = float(sys.argv[2]); arm = sys.argv[3]; maxf = int(sys.argv[4])
prob = A.Problem(d, 0, 9)
t = time.perf_counter(); ev = prob.evaluate(dict(prob.z0, e=prob.z0["e"])); t1 = time.perf_counter() - t
print(f"d={d} one evaluation {t1:.3f}s obj={ev['obj']:.6e} res0={A.canonical_residual(prob.z0, ev['grad']):.3e}")
g = ev["grad"]; print("grad norms beta %.2e t %.2e e %.2e c %.2e" % tuple(np.linalg.norm(g[k]) for k in ("beta","t","e","c")))
t = time.perf_counter()
if arm.startswith("pg"):
    out = A.run_pg(prob, arm[-1], 1e-7, maxf, alpha0)
else:
    out = A.run_lbfgs(prob, arm[-1], 1e-7, maxf)
el = time.perf_counter() - t
tr = out["trace"]
print(f"{arm} alpha0={alpha0} stop={out['stop']} fits={prob.fits} iters={tr[-1]['iteration']} time={el:.1f}s per-fit={el/prob.fits*1e3:.1f}ms")
for r in tr[::max(1,len(tr)//15)] + [tr[-1]]:
    print(f"  it={r['iteration']:5d} fits={r['fits']:5d} res={r['residual']:.3e} E~{r['rel_error_double']:.2e} t={['%.1e'%x for x in r['t']]}")
if out.get("alphas"):
    a = np.array(out["alphas"]); print("alphas: min %.3g max %.3g frac<alpha0 %.2f" % (a.min(), a.max(), (a < alpha0).mean()))
print("separation margin", out["ext"].separation(), "clipped", out["ext"].clipped)
