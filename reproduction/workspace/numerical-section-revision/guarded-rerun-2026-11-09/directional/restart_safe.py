"""Dormant restart-failure repair derived from the frozen guarded solver.

Finite successful numerical paths are unchanged. A tangent-frame roundtrip at
an exact admissibility boundary may fail its restart check by one ulp. Count
that rejected restart as a local return, retry once, and then return the last
admissible physical state with explicit status ``guarded_restart``.
"""
from pathlib import Path
import hashlib
import math
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np
from scipy.optimize import minimize
import chain_solver as frozen
from chain_solver import Geometry, StructureChange, WorkLimit, InadmissibleChart

FrozenSolver = frozen.Solver
VERSION = "restart-safe-v1"


class GuardedRestartReturn(Exception):
    """Two rejected restart attempts; the stored physical state is unchanged."""


class RestartSafeSolver(FrozenSolver):
    def _rejected_restart(self, exception):
        self.guard_rejection("restart", exception)
        self.guarded_restart_returns = getattr(self, "guarded_restart_returns", 0) + 1

    def fit_current(self):
        for _ in range(2):
            try:
                return super().fit_current()
            except InadmissibleChart as exception:
                self._rejected_restart(exception)
        raise GuardedRestartReturn

    def restart_geometry(self, first_step=0.01):
        for _ in range(2):
            self.warm = self.coef.copy()
            try:
                unit = Geometry(self.blocks, self.d, self.sigma, self.box)
            except InadmissibleChart as exception:
                self._rejected_restart(exception)
                continue
            self.cache = None
            objective, grad = self.evaluate(unit.x, unit)
            if not np.isfinite(objective):
                self._rejected_restart(InadmissibleChart("invalid_restart"))
                continue
            gnorm = float(np.linalg.norm(grad))
            scale = float(np.clip(math.sqrt(first_step * self.sigma / max(gnorm, 1e-300)), 1e-3, 1e3))
            try:
                geometry = Geometry(self.blocks, self.d, self.sigma, self.box, scale)
            except InadmissibleChart as exception:
                self._rejected_restart(exception)
                continue
            self.cache["p"], self.cache["grad"] = geometry.x.copy(), grad * scale
            return geometry
        raise GuardedRestartReturn

    def run(self):
        self.t0 = time.perf_counter()
        stop = "work_limit"
        try:
            self.fit_current(); self.record()
            self.scan()
            stalls = 0
            while self.fits < self.max_fits:
                geometry = self.restart_geometry()

                def callback(x):
                    obj, _ = self.evaluate(x, geometry)
                    if not np.isfinite(obj):
                        return
                    self.accept_cached()
                    self.iterations += 1
                    if self.iterations % self.scan_every == 0:
                        self.record()
                        if self.scan():
                            raise StructureChange
                try:
                    result = minimize(lambda p: self.evaluate(p, geometry), geometry.x, jac=True, method="L-BFGS-B",
                                      bounds=geometry.bounds, callback=callback,
                                      options={"ftol": 1e-15, "gtol": 1e-12, "maxls": 40, "maxiter": 6000})
                except StructureChange:
                    stalls = 0
                    continue
                obj, _ = self.evaluate(result.x, geometry)
                if np.isfinite(obj):
                    self.accept_cached()
                else:
                    # Keep the last accepted feasible state, never an invalid optimizer return.
                    self.fit_current()
                self.record()
                if self.method == "standard":
                    stop = "abnormal" if result.status == 2 else "function_tolerance"
                    break
                if self.scan():
                    stalls = 0
                    continue
                stalls += 1
                if stalls >= 2:
                    stop = "abnormal" if result.status == 2 else "function_tolerance"
                    break
        except GuardedRestartReturn:
            stop = "guarded_restart"
        except WorkLimit:
            stop = "work_limit"
        except FloatingPointError as exc:
            stop = str(exc)
        self.solve_seconds = time.perf_counter() - self.t0
        return self.summary(stop)

    def summary(self, stop):
        result = super().summary(stop)
        result["guarded_restart_returns"] = getattr(self, "guarded_restart_returns", 0)
        result["derived_solver_version"] = VERSION
        result["derived_solver_sha256"] = {"restart_safe.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        return result
