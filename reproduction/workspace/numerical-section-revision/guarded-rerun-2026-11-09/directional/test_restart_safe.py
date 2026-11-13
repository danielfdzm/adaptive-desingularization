"""Boundary regression and successful-path comparison for the derived solver."""
from pathlib import Path
import json
import hashlib
import unittest
from unittest.mock import patch
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np
import chain_experiments as ce
from restart_safe import FrozenSolver, RestartSafeSolver, GuardedRestartReturn, VERSION
import restart_safe

EVIDENCE = {}


class RestartTests(unittest.TestCase):
    def make_boundary_solver(self, cls):
        fixture = json.loads((HERE / "restart-boundary-regression.json").read_text())
        block = {key: np.asarray(value, float) if isinstance(value, list) else value
                 for key, value in fixture["block"].items()}
        target = (np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([0]), np.array([1.]))
        solver = cls(3, 1., target, [np.zeros(3)], "stable", np.zeros((1, 3)), np.ones(1))
        solver.blocks = [block]
        solver.coef = np.array([1., .1, .01])
        return solver, fixture["block"]

    def test_frozen_restart_bug_is_reproduced(self):
        solver, _ = self.make_boundary_solver(FrozenSolver)
        with self.assertRaises(TypeError):
            solver.restart_geometry()
        self.assertEqual(solver.fits, 0)

    def test_two_rejected_restarts_retain_state(self):
        solver, expected = self.make_boundary_solver(RestartSafeSolver)
        result = solver.run()
        self.assertEqual(result["stop"], "guarded_restart")
        self.assertEqual(result["guarded_restart_returns"], 2)
        self.assertEqual(result["fits"], 0)
        self.assertEqual(result["final_blocks"], [expected])
        self.assertEqual(result["final_coefficients"], [1., .1, .01])
        self.assertTrue(np.isfinite(result["final_relative_error"]))
        EVIDENCE["boundary"] = {"stop": result["stop"], "guarded_restart_returns": result["guarded_restart_returns"],
                                "fits": result["fits"], "state_preserved": result["final_blocks"] == [expected],
                                "coefficients_preserved": result["final_coefficients"] == [1., .1, .01]}

    def test_restart_geometry_has_explicit_failure(self):
        solver, expected = self.make_boundary_solver(RestartSafeSolver)
        with self.assertRaises(GuardedRestartReturn):
            solver.restart_geometry()
        self.assertEqual(solver.guarded_restart_returns, 2)
        self.assertEqual(solver.fits, 0)
        np.testing.assert_array_equal(solver.coef, [1., .1, .01])

    def test_scaled_geometry_construction_is_covered(self):
        problem = ce.problem_second1d(40)
        solver = RestartSafeSolver(problem["d"], problem["sigma"], problem["target"], problem["starts"],
                                   "stable", problem["X"], problem["W"], max_fits=problem["max_fits"], box=problem["box"])
        solver.fit_current()
        before_blocks = [dict(b, c=b["c"].copy()) for b in solver.blocks]
        before_coef = solver.coef.copy()
        constructor = restart_safe.Geometry
        def rejected_scaled_geometry(*args, **kwargs):
            if len(args) > 4 or "scale" in kwargs:
                raise restart_safe.InadmissibleChart("forced_scaled_geometry_rejection")
            return constructor(*args, **kwargs)
        with patch("restart_safe.Geometry", side_effect=rejected_scaled_geometry):
            with self.assertRaises(GuardedRestartReturn):
                solver.restart_geometry()
        self.assertEqual(solver.guarded_restart_returns, 2)
        np.testing.assert_array_equal(solver.coef, before_coef)
        for b, previous in zip(solver.blocks, before_blocks):
            np.testing.assert_array_equal(b["c"], previous["c"])
        EVIDENCE["second_geometry_constructor_covered"] = True

    def test_successful_full_runs_match_frozen(self):
        comparisons = []
        for method in ("stable", "adaptive"):
            problem = ce.problem_second1d(40)
            solver = RestartSafeSolver(problem["d"], problem["sigma"], problem["target"], problem["starts"],
                                       method, problem["X"], problem["W"], max_fits=problem["max_fits"], box=problem["box"])
            result = solver.run()
            original = json.loads((HERE / "accuracy-second1d" / f"second1d_d1_g0_seed40_{method}.json").read_text())
            fields = ("final_blocks", "final_coefficients", "events", "fits", "iterations", "joins", "completions", "final_relative_error")
            checks = {key: result[key] == original[key] for key in fields}
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(result["guarded_restart_returns"], 0)
            comparisons.append({"method": method, "seed": 40, "matches_frozen_accuracy": checks})
        (HERE / "restart-safe-comparison.json").write_text(json.dumps({"all_exactly_equal": True, "comparisons": comparisons}, indent=2) + "\n")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RestartTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "test_success": result.wasSuccessful(),
              "derived_solver_version": VERSION,
              "derived_solver_sha256": {"restart_safe.py": hashlib.sha256((HERE / "restart_safe.py").read_bytes()).hexdigest()},
              "frozen_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / "src").glob("*.py"))},
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              **EVIDENCE,
              "failures": [str(test) + "\n" + details for test, details in result.failures],
              "errors": [str(test) + "\n" + details for test, details in result.errors]}
    (HERE / "restart-safe-regression-report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
