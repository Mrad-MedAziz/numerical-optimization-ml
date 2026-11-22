import importlib.util
import unittest
import numpy as np
from optimization.separable_logistic import logistic_loss, logistic_gradient, max_margin_solution


class SeparableChecks(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        X = np.array([[2., -1.], [-3., 2.], [1., 4.]])
        y = np.array([1., -1., 1.])
        w = np.array([0.3, -0.2])
        h = 1e-5
        numerical = np.zeros(2)
        for j in range(2):
            change = np.zeros(2)
            change[j] = h
            numerical[j] = (logistic_loss(X, y, w + change)
                            - logistic_loss(X, y, w - change)) / (2 * h)
        np.testing.assert_allclose(logistic_gradient(X, y, w), numerical,
                                   rtol=1e-6, atol=1e-8)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Optional Gurobi dependency is not installed")
    def test_qp_matches_exact_four_point_solution(self):
        X = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
        y = np.array([1., -1., 1., -1.])
        w, dual = max_margin_solution(X, y)
        np.testing.assert_allclose(w, [1., 0.], atol=1e-6, rtol=0)
        self.assertGreaterEqual(np.min(y * (X @ w)), 1 - 1e-6)
        self.assertGreaterEqual(np.min(dual), -1e-6)


if __name__ == "__main__":
    unittest.main()
