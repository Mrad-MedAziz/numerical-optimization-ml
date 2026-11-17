"""Numerical regression checks. Run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from optimization import LinearRegression, LogisticRegression
from optimization.base import add_bias, backtracking_line_search
from experiments.study import generate_data


class NumericalChecks(unittest.TestCase):
    def test_capped_fits_log_the_returned_parameters(self):
        X, y = generate_data(42)
        for solver, kwargs in [("fit_gradient_ascent", {"epochs": 2}),
                               ("fit_newton", {"max_iter": 2})]:
            model = LogisticRegression()
            getattr(model, solver)(X, y, tol=0.0, **kwargs)
            self.assertEqual(model.n_iter_, 2)
            self.assertEqual(len(model.log_likelihood_history), 3)
            self.assertFalse(model.converged_)
            actual = model._log_likelihood(y, add_bias(X) @ model.theta)
            self.assertAlmostEqual(actual, model.log_likelihood_history[-1], places=14)

    def test_zero_update_budget_is_well_defined(self):
        X, y = generate_data(42)
        for solver, kwargs in [("fit_gradient_ascent", {"epochs": 0}),
                               ("fit_newton", {"max_iter": 0})]:
            model = LogisticRegression()
            getattr(model, solver)(X, y, **kwargs)
            self.assertEqual(model.n_iter_, 0)
            self.assertEqual(len(model.log_likelihood_history), 1)
            np.testing.assert_array_equal(model.theta, np.zeros(3))

    def test_gradient_matches_finite_differences(self):
        X, y = generate_data(4)
        Xb = add_bias(X)
        theta = np.array([0.3, -0.2, 0.7])
        model = LogisticRegression()
        gradient = Xb.T @ (y - model._sigmoid(Xb @ theta)) / len(y)
        numerical = []
        for i in range(len(theta)):
            delta = np.zeros_like(theta)
            delta[i] = 1e-6
            numerical.append((model._log_likelihood(y, Xb @ (theta + delta)) -
                              model._log_likelihood(y, Xb @ (theta - delta))) / 2e-6)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)

    def test_extreme_logits_remain_finite_and_unclipped(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            np.testing.assert_allclose(LogisticRegression._sigmoid(np.array([-1000., 0., 1000.])),
                                       [0., 0.5, 1.])
            value = LogisticRegression._log_likelihood(np.array([1., 0.]), np.array([-1000., 1000.]))
        self.assertEqual(value, -1000.)

    def test_damping_handles_duplicate_features(self):
        # A rank-deficient design gives singular undamped curvature. This is a
        # constructed numerical check, distinct from the original separability study.
        column = np.array([-2., -1., 0., 1., 2.])
        X = np.column_stack([column, column])
        y = np.array([0., 1., 0., 1., 1.])
        with self.assertRaises(np.linalg.LinAlgError):
            LogisticRegression().fit_newton(X, y, eps_reg=0.0)
        model = LogisticRegression()
        model.fit_newton(X, y, eps_reg=1e-8, max_iter=30)
        self.assertTrue(model.converged_)
        self.assertTrue(np.all(np.isfinite(model.theta)))

    def test_newton_matches_independent_gradient_ascent_solution(self):
        X, y = generate_data(7)
        a, b = LogisticRegression(), LogisticRegression()
        a.fit_newton(X, y, max_iter=30, tol=1e-10)
        b.fit_gradient_ascent(X, y, lr=2.0, epochs=10000, tol=1e-8)
        self.assertTrue(a.converged_ and b.converged_)
        np.testing.assert_allclose(a.theta, b.theta, atol=1e-5, rtol=1e-5)

    def test_ols_solvers_and_final_cost(self):
        X = np.array([[-2.], [-1.], [0.], [1.], [2.]])
        y = 1.5 + 2.0 * X[:, 0]
        exact = LinearRegression().fit_normal_equations(X, y)
        for solver in ["fit_gradient_descent", "fit_armijo_gradient_descent"]:
            model = LinearRegression()
            theta = getattr(model, solver)(X, y, epochs=1000)
            np.testing.assert_allclose(theta, exact, atol=1e-8)
            residual = model.predict(X) - y
            self.assertAlmostEqual(model.cost, residual @ residual / (2 * len(y)), places=14)

    def test_armijo_accepts_finite_decrease_and_rejects_failure(self):
        theta = np.array([2.])
        grad = 2 * theta
        cost = lambda t: float(t @ t)
        step = backtracking_line_search(cost, theta, grad)
        self.assertLessEqual(cost(theta - step * grad), cost(theta) - 1e-4 * step * (grad @ grad))
        with self.assertRaises(RuntimeError):
            backtracking_line_search(lambda _: float("nan"), theta, grad)
        with self.assertRaises(ValueError):
            backtracking_line_search(cost, theta, grad, tau=1.0)

    def test_invalid_binary_data_is_rejected(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit_newton(np.ones((2, 1)), np.array([0., 2.]))
        with self.assertRaises(ValueError):
            LogisticRegression().fit_newton(np.ones((2, 1)), np.array([[0.], [1.]]))


if __name__ == "__main__":
    unittest.main()
