#!/usr/bin/env python3
"""One first-order Poisson--Boltzmann recovery run (S, NC, C or T), Phase-3 cohort.

NC (`stable`) and C (`adaptive`) run the frozen guarded solver unchanged.
T installs the squared half-gap chart t = h^2 >= 0 from the unchanged
`squared_gap.py`. S runs ordinary Gaussians (`standard`) with a derived run
loop whose only change is the stopping rule: two local returns, as for NC, C
and T, instead of the first local return. See `derived_standard_run` below.
Optional --ftol/--gtol override the L-BFGS-B tolerances (used only by the
step-10 tolerance sweep; the cohort uses the frozen values).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SQDIR = ROOT / 'numerical-section-revision' / 'sinum-revision-2026-11-10' / 'squared-gap'
SRC = SQDIR / 'source'
sys.path.insert(0, str(SRC))


def derived_standard_run(self, run):
    """Frozen `Solver.run` with one change: S stops after two local returns.

    Lines marked CHANGED differ from the frozen source; S never scans or
    regroups, so a local return of S is never followed by a structural change.
    """
    np = run.np
    stop = "work_limit"
    try:
        self.fit_current()
        self.record()
        if self.method != "standard":
            self.regroup()
            self.fit_current()
            self.record()
        stalls = 0
        while self.fits < self.max_fits:
            geometry = run.Geometry(self.blocks, self.d)
            self.cache = None
            self.warm = self.coef.copy()

            def callback(x):
                self.evaluate(x, geometry)
                self.accept_cached()
                self.iterations += 1
                if self.iterations % 20 == 0:
                    self.record()
                    if self.method != "standard" and self.scan():
                        raise run.StructureChange
                if self.method == "standard" and self.cache["cond"] > 1e16:
                    raise FloatingPointError("ordinary Gram condition exceeds 1e16")
            try:
                result = run.minimize(lambda p: self.evaluate(p, geometry), geometry.x, jac=True, method="L-BFGS-B",
                                      bounds=geometry.bounds, callback=callback,
                                      options={"ftol": 1e-15, "gtol": 1e-12, "maxls": 40, "maxiter": 6000})
            except run.StructureChange:
                stalls = 0
                continue
            self.evaluate(result.x, geometry)
            self.accept_cached()
            self.record()
            termination = run.classify_termination(result, geometry.x, self.cache["gradient"], geometry.bounds)
            self.optimizer_returns.append({"fits": self.fits, **termination})
            if result.status == 2 or termination["classification"] == "numerical_stagnation":
                self.abnormal += 1
            if self.method != "standard" and self.scan():      # CHANGED: S has no scan
                stalls = 0
                continue
            stalls += 1
            if stalls >= 2:                                      # CHANGED: two local returns for S too
                stop = termination["classification"]
                break
    except run.WorkLimit:
        stop = "work_limit"
    except FloatingPointError as exc:
        stop = str(exc)
        self.abnormal += 1
    final = self.record()
    return self._final_payload(stop, final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dimension', type=int, required=True)
    ap.add_argument('--geometry', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--method', choices=['S', 'NC', 'C', 'T'], required=True)
    ap.add_argument('--max-fits', type=int, default=1500)
    ap.add_argument('--ftol', type=float)
    ap.add_argument('--gtol', type=float)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--context', default='accuracy campaign; several workers; times not used for comparisons')
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    import run
    import numpy as np
    import scipy
    import torch
    import mpmath
    from threadpoolctl import threadpool_info
    if args.method == 'T':
        spec = importlib.util.spec_from_file_location('squared_gap', SQDIR / 'squared_gap.py')
        squared_gap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(squared_gap)
        squared_gap.install(run)
    if args.ftol is not None or args.gtol is not None:       # step-10 sweep only
        frozen_minimize = run.minimize

        def minimize(*a, **kw):
            options = dict(kw.get('options', {}))
            if args.ftol is not None:
                options['ftol'] = args.ftol
            if args.gtol is not None:
                options['gtol'] = args.gtol
            kw['options'] = options
            return frozen_minimize(*a, **kw)
        run.minimize = minimize
    method = dict(S='standard', NC='stable', C='adaptive', T='squared_gap')[args.method]
    start = time.perf_counter()
    solver = run.Solver(args.dimension, args.seed, method, 12, args.max_fits, args.geometry, 8128, 1.)
    setup = time.perf_counter() - start
    solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    original_record = solver.record
    record_seconds = 0.

    def record(*a, **kw):
        nonlocal record_seconds
        t = time.perf_counter()
        try:
            return original_record(*a, **kw)
        finally:
            record_seconds += time.perf_counter() - t
    solver.record = record
    if args.method == 'S':
        # The frozen loop builds its payload inline; reuse it through a final-payload hook.
        frozen_run = run.Solver.run

        def final_payload(stop, final, self=solver):
            # Re-enter the frozen payload construction with the derived stop and final record.
            return payload_from_frozen(self, stop, final, run)
        solver._final_payload = final_payload
        result = derived_standard_run(solver, run)
        assert frozen_run is run.Solver.run
    else:
        result = solver.run()
    elapsed = time.perf_counter() - start
    audit_start = time.perf_counter()
    independent = run.high_precision_error(solver.blocks, solver.coef, solver.tb, solver.tc, dps=110)
    transfer_checks = []
    if args.method == 'C':
        from kernels import _mp_primitives, _mp_inner
        for event in result['events']:
            if event['kind'] != 'completion':
                continue
            i = next(k for k, b in enumerate(event['before_blocks']) if b['id'] == event['block_id'])
            ix = run.offsets(event['before_blocks'])
            c = event['before_coefficients'][ix[i]:ix[i + 1]]
            before_block = [event['before_blocks'][i]]
            trial_block = [event['trial_blocks'][i]]
            actual = run.high_precision_error(trial_block, c, before_block, c, dps=110)['norm']
            assert actual <= event['state_error_bound'] * (1 + 1e-8) + 1e-14 * solver.target2 ** .5
            assert actual <= event['state_tolerance'] * (1 + 1e-8)
            with mpmath.workdps(110):
                negate = lambda p: [(cc, [-x for x in v], -a) for cc, v, a in p]
                delta = _mp_primitives(trial_block, c) + negate(_mp_primitives(before_block, c))
                residual = _mp_primitives(event['before_blocks'], event['before_coefficients']) + negate(
                    _mp_primitives(solver.tb, solver.tc))
                quadratic_increment = float(_mp_inner(residual, delta, run.SIGMA) + _mp_inner(delta, delta, run.SIGMA) / 2)
            transfer_checks.append(dict(block_id=event['block_id'], fits=event['fits'],
                                        state_change_norm_110_digits=actual, state_bound_pass=True,
                                        quadratic_energy_increment_110_digits=quadratic_increment,
                                        recorded_gate=event['gate'], allowance=event['eps_j']))
    audit_seconds = time.perf_counter() - audit_start
    assert abs(independent['relative'] - result['final_relative_error']) <= max(1e-20, 1e-10 * independent['relative'])
    if args.method == 'T':
        result['zero_squared_gaps'] = sum(b.get('t') == 0. for b in solver.blocks if b['kind'] == 'pair')
        result['squared_gaps'] = [b.get('t', b['h'] ** 2) for b in solver.blocks if b['kind'] == 'pair']
    result['phase3_method'] = args.method
    result['guard_counts'] = solver.guard_counts
    result['independent_endpoint_check_110_digits'] = independent
    result['independent_transfer_checks'] = transfer_checks
    result['optimizer_overrides'] = dict(ftol=args.ftol, gtol=args.gtol) if (args.ftol is not None or args.gtol is not None) else None
    result['timing'] = dict(context=args.context, setup_seconds=setup, end_to_end_seconds=elapsed,
                            diagnostic_record_seconds=record_seconds,
                            end_to_end_without_diagnostic_records_seconds=elapsed - record_seconds,
                            independent_endpoint_audit_seconds=audit_seconds)
    paths = sorted(SRC.glob('*.py')) + [SQDIR / 'squared_gap.py', Path(__file__)]
    result['reproduction'] = dict(command=sys.argv, python=sys.version, platform=platform.platform(),
                                  numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                                  mpmath=mpmath.__version__, threadpools=threadpool_info(),
                                  torch_threads=torch.get_num_threads(), pid=os.getpid(),
                                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(run.plain(result), indent=2) + '\n')
    tmp.replace(args.output)
    print(json.dumps(dict(d=args.dimension, g=args.geometry, seed=args.seed, method=args.method,
                          error=result['final_relative_error'], fits=result['fits'], stop=result['stop'],
                          seconds=round(elapsed, 2))), flush=True)


def payload_from_frozen(self, stop, final, run):
    """The frozen `Solver.run` return payload, field for field."""
    import resource
    return run.plain({"dimension": self.d, "geometry_id": self.geometry_id, "seed": self.seed, "method": self.method,
                      "chi_nl_training": self.chi_nl, "target_sampled_peak": self.target_sampled_peak,
                      "setup_wall": self.setup_wall, "reaction_strength": self.reaction_strength,
                      "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      "nonlinear_value_calls": self.nonlinear_value_calls, "nonlinear_derivative_calls": self.nonlinear_derivative_calls,
                      "quadrature_point_evaluations": len(self.W) * (self.nonlinear_value_calls + self.nonlinear_derivative_calls),
                      "objective_normalization": "divide by target H1 norm squared",
                      "time_to_error": {str(t): next((r["wall"] for r in self.trace if r["relative_error"] <= t), None)
                                        for t in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)},
                      "initial_centers": self.starts, "blocks": self.blocks, "coefficients": self.coef,
                      "target_blocks": self.tb, "target_coefficients": self.tc, "C": self.C, "R_target": self.R,
                      "target_norm": math.sqrt(self.target2), "quadrature": self.qmeta,
                      "fits": self.fits, "iterations": self.iterations, "wall": time.perf_counter() - self.t0,
                      "stop": stop, "abnormal_returns": self.abnormal, "max_cond": self.max_cond,
                      "optimizer_status": self.optimizer_returns[-1]["status"] if self.optimizer_returns else None,
                      "optimizer_message": self.optimizer_returns[-1]["message"] if self.optimizer_returns else None,
                      "optimizer_returns": self.optimizer_returns,
                      "projected_gradient_max_abs": getattr(self, "accepted_projected_gradient", None),
                      "completions": self.completions, "transfer_rejections": self.rejections,
                      "best_relative_error": self.best_rel, "final_relative_error": final["relative_error"],
                      "coefficient_norm": final["coefficient_norm"], "energy_gap": final["energy_gap"],
                      "nonlinear_gap": final["nonlinear_gap"], "cost": final["cost"],
                      "coefficient_fit": getattr(self, "accepted_fit_info", None),
                      "trace": self.trace, "events": self.events,
                      "derived_stopping_rule": "two local returns (S); frozen loop otherwise unchanged"})


if __name__ == '__main__':
    main()
