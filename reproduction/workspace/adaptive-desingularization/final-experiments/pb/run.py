#!/usr/bin/env python3
"""Poisson--Boltzmann final suite in ambient dimensions 5, 10 and 20.

Run from the paper directory, e.g.
python3 pb-three-coordinate-directions/run.py --output pb-three-coordinate-directions/results --jobs 4

Every run starts with ten ordinary atoms. Adaptive grouping uses distances,
not the target's pair labels. No Tikhonov term is added to either objective.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.optimize import minimize
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
from fit import solve_coefficients, solve_unconstrained_quadratic
from kernels import block_gram, column_values, kappa, high_precision_error
from quadrature import make_quadrature, bregman_parts, weighted_coefficient_parts, weighted_bregman
from config import SIGMA, target_blocks, start_centers, configuration


def plain(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def arrays(blocks):
    result = copy.deepcopy(blocks)
    for b in result:
        b["c"] = np.asarray(b["c"], dtype=float)
        if b["kind"] == "pair":
            b["e"] = np.asarray(b["e"], dtype=float)
    return result


def offsets(blocks):
    return np.cumsum([0] + [1 if b["kind"] == "single" else 2 for b in blocks])


def projected_gradient_max(p, gradient, bounds):
    """Infinity norm of p - projection_box(p - gradient)."""
    lower, upper = np.asarray(bounds, dtype=float).T
    return float(np.max(np.abs(p - np.clip(p - gradient, lower, upper))))


def classify_termination(result, initial, gradient, bounds):
    """Describe an outer return without inferring stationarity from success.

    The stagnation threshold is diagnostic only; it does not alter the
    optimizer, accepted states, tolerances, or its subsequent restarts.
    """
    projected = projected_gradient_max(result.x, gradient, bounds)
    displacement = float(np.max(np.abs(result.x - initial)))
    message = str(result.message)
    if displacement == 0.0 and projected > 1e-6:
        label = "numerical_stagnation"
    elif result.success:
        if "PGTOL" in message.upper() or "PROJECTED_GRADIENT" in message.upper():
            label = "gradient_tolerance"
        elif "RELATIVE" in message.upper() or "FTOL" in message.upper():
            label = "function_tolerance"
        else:
            label = "optimizer_success"
    else:
        label = message
    return {"status": int(result.status), "message": message,
            "success": bool(result.success), "classification": label,
            "projected_gradient_max_abs": projected,
            "geometry_displacement_max_abs": displacement}


class Geometry:
    """Local tangent coordinates for unit directions; rebuild after a scan."""
    def __init__(self, blocks, d):
        self.blocks, self.d = arrays(blocks), d
        self.frames, vals, self.bounds = [], [], []
        for b in self.blocks:
            vals.extend(b["c"])
            self.bounds.extend([(-2.0, 2.0)] * d)
            if b["kind"] == "pair":
                self.frames.append(null_space(b["e"].reshape(1, d)))
                vals.extend([0.0] * (d - 1))
                self.bounds.extend([(-2.0, 2.0)] * (d - 1))
                if not b["completed"]:
                    vals.append(b["h"])
                    self.bounds.append((1e-12 * SIGMA, .6 * SIGMA))
            else:
                self.frames.append(None)
        self.x = np.array(vals)

    def unpack(self, p):
        blocks, j = [], 0
        for b, U in zip(self.blocks, self.frames):
            c = p[j:j+self.d]
            j += self.d
            new = {"id": b["id"], "kind": b["kind"], "c": c}
            if b["kind"] == "pair":
                e = torch.as_tensor(b["e"]) + torch.as_tensor(U) @ p[j:j+self.d-1]
                j += self.d - 1
                e = e / torch.linalg.vector_norm(e)
                new.update(e=e, completed=b["completed"], h=0.0)
                if not b["completed"]:
                    new["h"] = p[j]
                    j += 1
            blocks.append(new)
        return blocks


class WorkLimit(Exception):
    pass


class StructureChange(Exception):
    pass


class Solver:
    def __init__(self, d, seed, method, power, max_fits, geometry_id=0, quadrature_seed=8128, reaction_strength=1.0):
        self.t0 = time.perf_counter()
        self.d, self.seed, self.method, self.max_fits = d, seed, method, max_fits
        self.geometry_id, self.quadrature_seed = geometry_id, quadrature_seed
        self.reaction_strength = float(reaction_strength)
        self.tb, self.tc = target_blocks(d, geometry_id)
        self.tb = arrays(self.tb)
        self.tc = np.asarray(self.tc)
        self.target2 = float(self.tc @ block_gram(self.tb).numpy() @ self.tc)
        self.rho0, self.rho1 = kappa(d, 0), 2*kappa(d, 1)
        self.R = float(np.sum(self.rhos(self.tb)*np.abs(self.tc)))
        self.C = None
        self.X, self.W, self.qmeta = make_quadrature(np.array([b["c"] for b in self.tb]),
                                                   power=power, seed=quadrature_seed+100*geometry_id+d)
        self.W *= self.reaction_strength
        self.Xt = torch.as_tensor(self.X)
        self.U = column_values(self.tb, self.Xt).numpy() @ self.tc
        self.chi_nl = weighted_bregman(self.U, np.zeros_like(self.U), self.W) / (.5*self.target2)
        self.target_sampled_peak = float(np.max(np.abs(self.U)))
        self.starts = start_centers(d, seed, geometry_id)
        self.blocks = [{"id": str(i), "kind": "single", "c": c.copy()}
                       for i, c in enumerate(self.starts)]
        self.coef = np.zeros(10)
        self.fits, self.iterations, self.scans, self.allowance_index = 0, 0, 0, 0
        self.events, self.trace, self.flags = [], [], {}
        self.optimizer_returns = []
        self.cache, self.warm = None, None
        self.completions, self.rejections, self.abnormal = 0, 0, 0
        self.max_cond, self.best_rel = 0., math.inf
        self.setup_wall = time.perf_counter()-self.t0
        self.quadrature_nodes_total = 0
        self.nonlinear_derivative_calls = 0
        self.nonlinear_value_calls = 0

    def rhos(self, blocks):
        return np.array([r for b in blocks for r in
                         ([self.rho0] if b["kind"] == "single" else [self.rho0, self.rho1])])

    def matrices(self, blocks):
        M = block_gram(blocks)
        b = block_gram(blocks, self.tb) @ torch.as_tensor(self.tc)
        Phi = column_values(blocks, self.Xt)
        return M, b, Phi

    def fit_matrices(self, M, b, Phi, blocks):
        if self.fits >= self.max_fits:
            raise WorkLimit
        self.fits += 1
        M = (M+M.T)/2
        rho = self.rhos(blocks)
        C = None
        scale = np.sqrt(np.maximum(np.diag(M), 1e-300))
        Mn = M / scale[:, None] / scale[None, :]
        cond = float(np.linalg.cond(Mn))
        self.max_cond = max(self.max_cond, cond)
        if self.warm is not None and len(self.warm) == len(b):
            g0 = self.warm.copy()
        else:
            g0 = solve_unconstrained_quadratic(M, b)
        g, info = solve_coefficients(M/self.target2, b/self.target2, rho, C, Phi, self.U, self.W/self.target2, g0)
        self.nonlinear_derivative_calls += info.get("nonlinear_derivatives", 0)
        self.nonlinear_value_calls += info.get("nonlinearvals", 0)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("nonfinite coefficient fit")
        gap, residual, _ = bregman_parts(Phi @ g, self.U)
        nonlinear = float(self.W @ gap)
        objective = (.5*g@M@g-b@g + nonlinear)/self.target2  # target-only constant omitted
        if not math.isfinite(objective):
            raise FloatingPointError("nonfinite objective")
        return g, objective, residual, nonlinear, cond, info

    def evaluate(self, p, geometry):
        if self.cache is not None and np.array_equal(p, self.cache["p"]):
            return self.cache["objective"], self.cache["gradient"]
        pt = torch.tensor(p, requires_grad=True)
        blocks = geometry.unpack(pt)
        Mt, bt, Phit = self.matrices(blocks)
        M, b, Phi = (v.detach().numpy() for v in (Mt, bt, Phit))
        g, obj, residual, nonlinear, cond, info = self.fit_matrices(M, b, Phi, blocks)
        gt = torch.as_tensor(g)
        # Envelope derivative: the coefficient solve is not differentiated.
        proxy = .5*gt@Mt@gt-bt@gt + torch.as_tensor(self.W*residual) @ (Phit@gt)
        grad = torch.autograd.grad(proxy/self.target2, pt)[0].detach().numpy()
        self.cache = {"p": np.array(p), "blocks": arrays(plain(blocks)), "coef": g.copy(),
                      "objective": obj, "gradient": grad, "nonlinear": nonlinear,
                      "cond": cond, "fit_info": info,
                      "projected_gradient_max_abs": projected_gradient_max(p, grad, geometry.bounds)}
        return obj, grad

    def accept_cached(self):
        self.blocks = arrays(self.cache["blocks"])
        self.coef = self.cache["coef"].copy()
        # Rejected geometry trials must not seed later coefficient solves.
        # Their ordinary coefficients can be enormous near a collision.
        self.warm = self.coef.copy()
        self.accepted_fit_info = self.cache["fit_info"]
        self.accepted_projected_gradient = self.cache["projected_gradient_max_abs"]

    def fit_current(self):
        geo = Geometry(self.blocks, self.d)
        self.cache = None
        self.evaluate(geo.x, geo)
        self.accept_cached()

    def energy(self, blocks, coef, precise=False):
        if precise:
            e = high_precision_error(blocks, coef, self.tb, self.tc)["norm"]
            quadratic = .5*e*e
        else:
            M, b, _ = self.matrices(blocks)
            quadratic = .5*coef@M.numpy()@coef-coef@b.numpy()+.5*self.target2
        nonlinear = weighted_bregman(column_values(blocks, self.Xt).numpy()@coef, self.U, self.W)
        return float(quadratic+nonlinear), nonlinear

    def record(self):
        err = high_precision_error(self.blocks, self.coef, self.tb, self.tc)
        energy, nonlinear = self.energy(self.blocks, self.coef, precise=True)
        self.best_rel = min(self.best_rel, err["relative"])
        raw, halfgaps = [], []
        ix = offsets(self.blocks)
        for i,b in enumerate(self.blocks):
            g = self.coef[ix[i]:ix[i+1]]
            if b["kind"] == "single":
                raw.append(g[0])
            elif b["completed"]:
                raw.extend(g)
            else:
                h=b["h"]
                halfgaps.append(h)
                raw.extend([g[0]/2+g[1]/(2*h),g[0]/2-g[1]/(2*h)])
        if self.method == "standard":
            centers=np.array([b["c"] for b in self.blocks])
            D=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)
            np.fill_diagonal(D,np.inf)
            halfgaps=[float(D.min()/2)]
        row={"fits":self.fits,"iterations":self.iterations,"wall":time.perf_counter()-self.t0,
             "relative_error":err["relative"],"error_norm":err["norm"],
             "energy_gap":energy,"nonlinear_gap":nonlinear,"coefficient_norm":float(np.linalg.norm(self.coef)),
             "ordinary_finite_part_norm":float(np.linalg.norm(raw)),
             "min_positive_halfgap":min(halfgaps) if halfgaps else None,
             "cost":float(self.rhos(self.blocks)@np.abs(self.coef)),"completions":self.completions}
        M=block_gram(self.blocks).numpy()
        scale=np.sqrt(np.diag(M))
        row["stored_gram_condition"]=float(np.linalg.cond(M/scale[:,None]/scale[None,:]))
        self.trace.append(row)
        progress=os.environ.get("PB_PROGRESS_DIR")
        if progress:
            path=Path(progress)/f"d{self.d}_g{self.geometry_id}_seed{self.seed}_{self.method}.json"
            path.write_text(json.dumps(plain({**row,"dimension":self.d,"method":self.method,
                                             "seed":self.seed,"pid":os.getpid()}),indent=2)+"\n")
        return row

    def diagnostics(self, i):
        b = self.blocks[i]
        h = b["h"]
        cs = [dict(kind="single",c=b["c"]+h*b["e"]), dict(kind="single",c=b["c"]-h*b["e"])]
        M = block_gram(cs).numpy()
        q = abs(float(M[0,1]/math.sqrt(M[0,0]*M[1,1])))
        ratio = math.sqrt(max(0.,1-q))
        return ratio

    def regroup(self):
        changed=False
        while True:
            indices=[i for i,b in enumerate(self.blocks) if b["kind"]=="single"]
            candidates=sorted((np.linalg.norm(self.blocks[i]["c"]-self.blocks[j]["c"]),i,j)
                              for k,i in enumerate(indices) for j in indices[k+1:])
            success=False
            for dist,i,j in candidates:
                if dist>SIGMA:
                    break
                if dist==0:
                    continue
                ix=offsets(self.blocks)
                bi,bj=self.blocks[i],self.blocks[j]
                wi,wj=self.coef[ix[i]],self.coef[ix[j]]
                h=dist/2
                a,beta=wi+wj,h*(wj-wi)
                old=self.rho0*(abs(wi)+abs(wj))
                new=self.rho0*abs(a)+self.rho1*abs(beta)
                cost=self.rhos(self.blocks)@np.abs(self.coef)-old+new
                pair={"id":bi["id"]+":"+bj["id"],"kind":"pair","c":(bi["c"]+bj["c"])/2,
                      "e":(bj["c"]-bi["c"])/dist,"h":h,"completed":False}
                rest=[k for k in range(len(self.blocks)) if k not in (i,j)]
                newcoef=np.concatenate([*[self.coef[ix[k]:ix[k+1]] for k in rest],np.array([a,beta])])
                self.blocks=[self.blocks[k] for k in rest]+[pair]
                self.coef=newcoef
                self.events.append({"kind":"regroup","fits":self.fits,"block_id":pair["id"],
                                    "h":h,"old_cost":old,"new_cost":new,"exact":True})
                success=changed=True
                break
            if not success:
                break
        if changed:
            self.warm=self.coef.copy()
            self.cache=None
        return changed

    def transfer(self, i):
        b=self.blocks[i]
        ix=offsets(self.blocks)
        a,beta=self.coef[ix[i]:ix[i+1]]
        trial=arrays(self.blocks)
        trial[i].update(completed=True,h=0.)
        # An analytic remainder bound holds in every dimension, with no tail truncation.
        error_bound=b["h"]**2*(abs(a)*kappa(self.d,2)/2+abs(beta)*kappa(self.d,3)/6)
        norm=math.sqrt(max(float(self.coef@block_gram(self.blocks).numpy()@self.coef),0.))
        before_energy,_=self.energy(self.blocks,self.coef,precise=True)
        trial_energy,_=self.energy(trial,self.coef,precise=True)
        gate=trial_energy-before_energy
        eps=1e-3*self.target2*2.**(-self.allowance_index-1)
        event={"kind":"completion" if error_bound<=.01*norm and gate<=eps else "transfer_rejected",
               "fits":self.fits,"block_id":b["id"],"h":b["h"],"direction":b["e"].copy(),
               "state_error_bound":error_bound,"state_tolerance":.01*norm,"gate":gate,"eps_j":eps,
               "before_blocks":plain(self.blocks),"before_coefficients":self.coef.tolist(),
               "trial_blocks":plain(trial),"trial_coefficients":self.coef.tolist()}
        if event["kind"]=="transfer_rejected":
            self.rejections+=1
            self.events.append(event)
            return False
        self.blocks=trial
        self.allowance_index+=1
        self.completions+=1
        self.warm=self.coef.copy()
        self.events.append(event)
        self.fit_current()
        after_energy,_=self.energy(self.blocks,self.coef,precise=True)
        event.update(after_blocks=plain(self.blocks),after_coefficients=self.coef.tolist(),
                     dJ_after_refit=after_energy-before_energy)
        self.record()
        return True

    def scan(self):
        self.scans+=1
        changed=self.regroup()
        for i,b in enumerate(self.blocks):
            if b["kind"]!="pair" or b["completed"]:
                continue
            ratio=self.diagnostics(i)
            flag=2*b["h"]<=.5*SIGMA and ratio<=.1
            self.flags[b["id"]]=self.flags.get(b["id"],0)+1 if flag else 0
            self.events.append({"kind":"scan","fits":self.fits,"scan":self.scans,"block_id":b["id"],
                                "gram_smallest_singular_value":ratio,"flag":flag,
                                "persistence":self.flags[b["id"]],"h":b["h"]})
            if self.method == "adaptive" and self.flags[b["id"]]>=2 and self.transfer(i):
                return True
        return changed

    def run(self):
        stop="work_limit"
        try:
            self.fit_current()
            self.record()
            if self.method!="standard":
                self.regroup()
                self.fit_current()
                self.record()
            stalls=0
            while self.fits<self.max_fits:
                geometry=Geometry(self.blocks,self.d)
                self.cache=None
                self.warm=self.coef.copy()
                def callback(x):
                    self.evaluate(x,geometry)
                    self.accept_cached()
                    self.iterations+=1
                    if self.iterations%20==0:
                        self.record()
                        if self.method!="standard" and self.scan():
                            raise StructureChange
                    if self.method=="standard" and self.cache["cond"]>1e16:
                        raise FloatingPointError("ordinary Gram condition exceeds 1e16")
                try:
                    result=minimize(lambda p:self.evaluate(p,geometry),geometry.x,jac=True,method="L-BFGS-B",
                                    bounds=geometry.bounds,callback=callback,
                                    options={"ftol":1e-15,"gtol":1e-12,"maxls":40,"maxiter":6000})
                except StructureChange:
                    stalls=0
                    continue
                self.evaluate(result.x,geometry)
                self.accept_cached()
                self.record()
                termination=classify_termination(result,geometry.x,self.cache["gradient"],geometry.bounds)
                self.optimizer_returns.append({"fits":self.fits,**termination})
                if result.status==2 or termination["classification"]=="numerical_stagnation":
                    self.abnormal+=1
                if self.method=="standard":
                    stop=termination["classification"]
                    break
                if self.scan():
                    stalls=0
                    continue
                stalls+=1
                if stalls>=2:
                    stop=termination["classification"]
                    break
        except WorkLimit:
            stop="work_limit"
        except FloatingPointError as exc:
            stop=str(exc)
            self.abnormal+=1
        final=self.record()
        return plain({"dimension":self.d,"geometry_id":self.geometry_id,"seed":self.seed,"method":self.method,
                      "chi_nl_training":self.chi_nl,"target_sampled_peak":self.target_sampled_peak,
                      "setup_wall":self.setup_wall,"reaction_strength":self.reaction_strength,
                      "worker_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=="darwin" else 1024),
                      "nonlinear_value_calls":self.nonlinear_value_calls,"nonlinear_derivative_calls":self.nonlinear_derivative_calls,
                      "quadrature_point_evaluations":len(self.W)*(self.nonlinear_value_calls+self.nonlinear_derivative_calls),"objective_normalization":"divide by target H1 norm squared",
                      "time_to_error":{str(t):next((r["wall"] for r in self.trace if r["relative_error"]<=t),None) for t in (1e-3,1e-4,1e-5,1e-6,1e-7)},
                      "initial_centers":self.starts,"blocks":self.blocks,"coefficients":self.coef,
                      "target_blocks":self.tb,"target_coefficients":self.tc,"C":self.C,"R_target":self.R,
                      "target_norm":math.sqrt(self.target2),"quadrature":self.qmeta,
                      "fits":self.fits,"iterations":self.iterations,"wall":time.perf_counter()-self.t0,
                      "stop":stop,"abnormal_returns":self.abnormal,"max_cond":self.max_cond,
                      "optimizer_status":self.optimizer_returns[-1]["status"] if self.optimizer_returns else None,
                      "optimizer_message":self.optimizer_returns[-1]["message"] if self.optimizer_returns else None,
                      "optimizer_returns":self.optimizer_returns,
                      "projected_gradient_max_abs":getattr(self,"accepted_projected_gradient",None),
                      "completions":self.completions,"transfer_rejections":self.rejections,
                      "best_relative_error":self.best_rel,"final_relative_error":final["relative_error"],
                      "coefficient_norm":final["coefficient_norm"],"energy_gap":final["energy_gap"],
                      "nonlinear_gap":final["nonlinear_gap"],"cost":final["cost"],
                      "coefficient_fit":getattr(self,"accepted_fit_info",None),
                      "trace":self.trace,"events":self.events})


def run_one(task):
    d,g,seed,method,power,max_fits,qseed,reaction=task
    return Solver(d,seed,method,power,max_fits,g,qseed,reaction).run()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions",nargs="+",type=int,default=[5,10,20])
    parser.add_argument("--seeds",nargs=2,type=int,default=[10,15],help="half-open seed interval")
    parser.add_argument("--methods",nargs="+",choices=["standard","stable","adaptive"],default=["standard","stable","adaptive"])
    parser.add_argument("--power",type=int,default=10)
    parser.add_argument("--geometries",nargs="+",type=int,default=[0,1])
    parser.add_argument("--quadrature-seed",type=int,default=8128)
    parser.add_argument("--reaction-strength",type=float,default=1.)
    parser.add_argument("--max-fits",type=int,default=1500)
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit("Output exists; use a new directory or --resume.")
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=[(d,g,s,m,args.power,args.max_fits,args.quadrature_seed,args.reaction_strength) for d in args.dimensions
           for g in args.geometries for s in range(*args.seeds) for m in args.methods]
    t0=time.perf_counter()
    sources=[HERE/n for n in ("run.py","fit.py","kernels.py","quadrature.py","config.py")]
    provenance={"command":sys.argv,"python":platform.python_version(),"numpy":np.__version__,
                "scipy":scipy.__version__,"torch":torch.__version__,"platform":platform.platform(),
                "threads_per_worker":1,"jobs":args.jobs,"coefficient_constraint":None,"coefficient_solver":"full-rank scaled Newton; no ridge or rank truncation","max_fits":args.max_fits,"power":args.power,
                "dimensions":args.dimensions,"geometries":args.geometries,"quadrature_seed":args.quadrature_seed,"reaction_strength":args.reaction_strength,
                "seeds":args.seeds,"methods":args.methods,"pid":os.getpid(),
                "source_hashes":{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "configuration":{f"d{d}_g{g}":configuration(d,g) for d in args.dimensions for g in args.geometries}}
    provenance_file=args.output/"provenance.json"
    if args.resume and provenance_file.exists():
        old=json.loads(provenance_file.read_text())
        if old["source_hashes"]!=provenance["source_hashes"]:
            raise SystemExit("Cannot resume with changed solver sources.")
        for setting in ("power","max_fits","dimensions","geometries","quadrature_seed","reaction_strength","seeds","methods"):
            if old.get(setting)!=provenance[setting]:
                raise SystemExit(f"Cannot resume with changed {setting}.")
    provenance_file.write_text(json.dumps(plain(provenance),indent=2)+"\n")
    def path(task):
        return args.output/f"d{task[0]}_g{task[1]}_seed{task[2]}_{task[3]}.json"
    pending=[t for t in tasks if not path(t).exists()]
    progress_dir=args.output/"progress"
    progress_dir.mkdir(exist_ok=True)
    os.environ["PB_PROGRESS_DIR"]=str(progress_dir.resolve())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_one,t):t for t in pending}
        for future in as_completed(futures):
            task=futures[future]
            result=future.result()
            path(task).write_text(json.dumps(result,indent=2)+"\n")
            print(f"d={task[0]} geometry={task[1]} seed={task[2]} {task[3]} error={result['final_relative_error']:.3e} "
                  f"fits={result['fits']} completions={result['completions']} wall={result['wall']:.1f}s "
                  f"stop={result['stop']}",flush=True)
    results=[json.loads(path(t).read_text()) for t in tasks]
    (args.output/"results.json").write_text(json.dumps({"runs":results},indent=2)+"\n")
    provenance["elapsed_seconds"]=time.perf_counter()-t0
    provenance_file.write_text(json.dumps(plain(provenance),indent=2)+"\n")


if __name__=="__main__":
    main()
