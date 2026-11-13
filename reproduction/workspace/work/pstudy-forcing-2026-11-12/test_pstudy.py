import sys, time, json
import numpy as np
sys.path.insert(0, ".")
import pstudy as ps
import chain_solver as cs
for case in ps.CASES:
    t = time.perf_counter(); chk = ps.reference_check(case)
    print(case, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in chk.items()}, f"{time.perf_counter()-t:.1f}s")
# finite-difference gradient check of the forcing objective (NC, P=4)
s = ps.ForcingSolver("sech_dipole", 4, 1, "NC", max_fits=10**6)
s.fit_current(); s.regroup(); s.fit_current()
g = cs.Geometry(s.blocks, 1, s.sigma, s.box)
x = g.x.copy(); s.cache = None; val, grad = s.evaluate(x, g); fd = []
for i in range(len(x)):
    h = 1e-6; xp = x.copy(); xm = x.copy(); xp[i] += h; xm[i] -= h
    s.cache = None; fp = s.evaluate(xp, g)[0]; s.cache = None; fm = s.evaluate(xm, g)[0]
    fd.append((fp - fm) / (2 * h))
print("blocks", s.order_counts(), "grad check max abs diff", float(np.max(np.abs(np.array(fd) - grad))), "grad norm", float(np.linalg.norm(grad)))
for m in ("S", "NC", "C"):
    t = time.perf_counter(); r = ps.run_one("single_source", 2, 300, m)
    print(m, r.get("stop"), "fits", r.get("fits"), "err", r.get("relative_H1_error"), "E", r.get("final_energy_normalized"),
          "coll", r.get("collision"), "blocks", r.get("blocks_final"), f"{time.perf_counter()-t:.1f}s")
print("galerkin P=2", ps.gaussian_galerkin("single_source", 2))
print("fem n=8", ps.fem_p1("single_source", 8)); print("fem n=64", ps.fem_p1("single_source", 64))
