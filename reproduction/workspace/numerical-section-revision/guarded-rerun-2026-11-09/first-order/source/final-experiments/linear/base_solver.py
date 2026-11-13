"""Shared geometry and exact pair regrouping from the corrected PB benchmark."""
import copy, math
from pair_guard import eligible_pair
import numpy as np
import torch
from scipy.linalg import null_space
from kernels import block_gram, kappa
SIGMA=.45

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

class PairOperations:
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
                    if not eligible_pair(self, dist, i, j, SIGMA):
                        continue
                    if dist>SIGMA:
                        break
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
