"""Adversarial and state-preservation checks for the new C.4 implementation."""
import unittest
import numpy as np
import torch

import chain_solver as cs
from directional_guard import InadmissibleChart, guard_offsets, guard_stored


class GuardTests(unittest.TestCase):
    def test_transverse_counterexample(self):
        for s in (1e-2, 1e-4, 1e-6):
            with self.assertRaisesRegex(InadmissibleChart, "transverse_cap"):
                guard_offsets([0, 0, s], [[0, s / 2]], 2, 1.)

    def test_zero_unresolved_and_nonfinite(self):
        for s in (0., 1e-16):
            with self.assertRaisesRegex(InadmissibleChart, "unresolved_joined_separation"):
                guard_offsets([0, 0, s], [[0, 0]], 2, 1.)
        with self.assertRaises(InadmissibleChart):
            guard_offsets([0, 0, np.nan], [[0, 0]], 2, 1.)

    def test_repeated_base_is_valid_and_scale_invariant(self):
        for sigma in (.1, 1., 10.):
            products, cap = guard_offsets(np.array([0, 0, .2]) * sigma,
                                         [[0, .001 * sigma]], 2, sigma)
            np.testing.assert_allclose(products, [.04])
            self.assertAlmostEqual(cap, .2)

    def make_solver(self):
        target = (np.zeros((1, 2)), np.array([[1., 0.]]), np.array([0]), np.array([1.]))
        return cs.Solver(2, .7, target, [np.array([.1, .1])], "stable",
                         np.array([[0., 0.], [.2, .1], [-.2, -.1]]), np.ones(3), max_fits=20)

    def test_successive_join_preserves_state(self):
        solver = self.make_solver(); sg = solver.sigma
        b = {"kind": "chain", "k": 1, "c": np.zeros(2), "e": np.array([1., 0.]), "id": 1}
        g = np.array([.8, -.3])
        for s, q, weight in ((.2, .001, .4), (.35, -.0001, -.7)):
            atom = np.array([s, q]) * sg
            old = [b, {"kind": "atom", "c": atom, "id": 9}]
            old_g = np.r_[g, weight]
            new, new_g, _, _ = solver.join(b, g, atom, weight)
            X = torch.tensor(np.random.default_rng(8).normal(size=(40, 2)) * sg)
            def values(blocks, coefs):
                C, U, O, WS = cs.primitives(blocks, 2, sg)
                return (cs.ck.point_values(X, C, U, O, sg) @ WS @ torch.tensor(coefs)).numpy()
            np.testing.assert_allclose(values(old, old_g), values([new], new_g), rtol=2e-10, atol=2e-11)
            guard_stored(cs.block_nodes(new), new["p"], cs.base_size(new), sg)
            b, g = new, new_g

    def test_enlarged_join_rechecks_existing_offsets(self):
        solver = self.make_solver(); sg = solver.sigma
        b = {"kind": "chain", "k": 1, "c": np.zeros(2), "e": np.array([1., 0.]), "id": 1}
        b, g, _, _ = solver.join(b, np.array([1., .2]), np.array([.2, .002]) * sg, .4)
        original_p = b["p"].copy()
        with self.assertRaisesRegex(InadmissibleChart, "transverse_cap"):
            solver.join(b, g, np.array([.3, 0.]) * sg, .5)
        np.testing.assert_array_equal(b["p"], original_p)

    def test_interior_geometry_gradient(self):
        solver = self.make_solver(); sg = solver.sigma
        b = {"kind": "line", "base": "chain", "kb": 1, "k": 3, "id": 1,
             "c": np.array([.03, -.02]), "e": np.array([1., 0.]),
             "s": np.array([.2, .35]) * sg, "p": np.array([[0., .02], [0., -.01]])}
        geometry = cs.Geometry([b], 2, sg, 2.)
        coefficients = torch.tensor([.8, -.3, .04, -.006])
        def scalar(parameters):
            blocks = geometry.unpack(parameters)
            M, rhs, Phi = cs.assemble(blocks, 2, sg, solver.X, solver.target)
            values = Phi @ coefficients
            return .5 * coefficients @ M @ coefficients - rhs @ coefficients + .1 * (values ** 2).sum()
        p = torch.tensor(geometry.x, requires_grad=True)
        analytic = torch.autograd.grad(scalar(p), p)[0].numpy()
        numerical = np.zeros_like(analytic)
        for i in range(len(numerical)):
            plus, minus = geometry.x.copy(), geometry.x.copy()
            plus[i] += 1e-6; minus[i] -= 1e-6
            numerical[i] = (float(scalar(torch.tensor(plus))) - float(scalar(torch.tensor(minus)))) / 2e-6
        np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-8)

    def test_invalid_optimizer_trial_precedes_fit(self):
        solver = self.make_solver(); sg = solver.sigma
        b = {"kind": "chain", "k": 1, "c": np.zeros(2), "e": np.array([1., 0.]), "id": 1}
        b, g, _, _ = solver.join(b, np.array([1., .2]), np.array([.2, 0.]) * sg, .4)
        solver.blocks = [b]; solver.coef = g
        geometry = cs.Geometry(solver.blocks, 2, sg, 2.)
        p = geometry.x.copy(); p[-1] = .9  # stored p-hat exceeds rho=.2
        before = solver.fits
        objective, gradient = solver.evaluate(p, geometry)
        self.assertTrue(np.isinf(objective)); self.assertEqual(solver.fits, before)
        self.assertIsNone(solver.cache)
        np.testing.assert_array_equal(gradient, np.zeros_like(p))
        self.assertEqual(solver.guard_counts["optimizer_trial:stored_transverse_cap"], 1)


if __name__ == "__main__":
    unittest.main()
