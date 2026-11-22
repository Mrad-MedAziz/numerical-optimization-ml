import importlib.util
import unittest
import numpy as np
from optimization.separable_logistic import (logistic_loss, logistic_gradient, max_margin_solution,
                                             regularized_loss, regularized_gradient, gradient_diagnostic)
from experiments.separable import run_one
from experiments.separable_extensions import run_regularized, regularized_reference


class SeparableChecks(unittest.TestCase):
    def test_threshold_crossings_and_measurements_match_actual_iterates(self):
        X = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
        y = np.array([1., -1., 1., -1.])
        w_hat = np.array([1., 0.])
        lr = 0.1
        weights = [np.zeros(2)]
        gradients = []
        for step in range(9):
            w = weights[-1]
            # Independent scalar-observation calculation of each coordinate.
            gradient = sum(-label * point / (1 + np.exp(label * (point @ w)))
                           for point, label in zip(X, y)) / len(y)
            gradients.append(gradient)
            if step < 8:
                weights.append(w - lr * gradient)
        norms = np.linalg.norm(gradients, axis=1)
        threshold = (norms[3] + norms[4]) / 2
        history, stops, final = run_one(X, y, w_hat, lr, {}, n_steps=8,
                                        tolerances=[1.0, threshold, 0.0])
        by_tolerance = {row["tolerance"]: row for row in stops}
        self.assertEqual(by_tolerance[1.0]["update"], 0)
        self.assertEqual(by_tolerance[threshold]["update"], 4)
        self.assertTrue(np.all(norms[:4] >= threshold))
        self.assertLess(norms[4], threshold)
        self.assertEqual(by_tolerance[threshold]["stop_reason"], "gradient_tolerance")
        self.assertEqual(by_tolerance[0.0]["stop_reason"], "iteration_budget")
        self.assertEqual(by_tolerance[0.0]["update"], 8)
        for row in history + stops + [final]:
            step = row["update"]
            w = weights[step]
            norm = np.linalg.norm(w)
            signed = np.array([label * (point @ w) for point, label in zip(X, y)])
            expected_loss = sum(np.log1p(np.exp(-score)) for score in signed) / len(y)
            self.assertAlmostEqual(row["loss"], expected_loss, places=14)
            self.assertAlmostEqual(row["gradient_norm"], norms[step], places=14)
            self.assertAlmostEqual(row["weight_norm"], norm, places=14)
            self.assertAlmostEqual(row["minimum_signed_score"], signed.min(), places=14)
            if step == 0:
                self.assertTrue(np.isnan(row["direction_error"]))
                self.assertTrue(np.isnan(row["geometric_margin"]))
            else:
                self.assertAlmostEqual(row["direction_error"], np.linalg.norm(w / norm - w_hat), places=14)
                self.assertAlmostEqual(row["geometric_margin"], signed.min() / norm, places=14)
                self.assertAlmostEqual(row["margin_ratio"], signed.min() / norm, places=14)

    def test_zero_budget_records_initial_state_as_budget_stop(self):
        X = np.array([[1., 0.], [-1., 0.]])
        y = np.array([1., -1.])
        history, stops, final = run_one(X, y, np.array([1., 0.]), 0.1, {},
                                        n_steps=0, tolerances=[0.0])
        self.assertEqual(len(history), 1)
        self.assertEqual(final["update"], 0)
        self.assertEqual(stops[0]["stop_reason"], "iteration_budget")
        self.assertAlmostEqual(final["loss"], np.log(2))

    def test_grouped_contributions_reconstruct_gradient_and_trial_updates(self):
        X = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
        y = np.array([1., -1., 1., -1.])
        w = np.array([0.8, 0.4])
        result = gradient_diagnostic(X, y, w, np.array([1., 0.]))
        self.assertEqual(result["support_indices"], [0, 1])
        grouped = np.array(result["support_gradient"]) + result["other_gradient"]
        np.testing.assert_allclose(grouped, logistic_gradient(X, y, w), atol=1e-14)
        for name, key in [("support_only", "support_gradient"), ("other_only", "other_gradient"),
                           ("combined", "full_gradient")]:
            trial = w - 0.1 * np.array(result[key])
            error = np.linalg.norm(trial / np.linalg.norm(trial) - np.array([1., 0.]))
            self.assertAlmostEqual(result["trial_direction_errors"][name], error, places=14)

    def test_regularized_gradient_and_zero_penalty(self):
        X = np.array([[2., -1.], [-3., 2.], [1., 4.]])
        y = np.array([1., -1., 1.])
        w = np.array([0.3, -0.2])
        penalty = 0.01
        h = 1e-5
        numerical = []
        for delta in np.eye(2) * h:
            numerical.append((regularized_loss(X, y, w + delta, penalty)
                              - regularized_loss(X, y, w - delta, penalty)) / (2 * h))
        np.testing.assert_allclose(regularized_gradient(X, y, w, penalty), numerical,
                                   rtol=1e-6, atol=1e-8)
        self.assertEqual(regularized_loss(X, y, w, 0), logistic_loss(X, y, w))
        np.testing.assert_array_equal(regularized_gradient(X, y, w, 0), logistic_gradient(X, y, w))

    def test_regularized_fit_matches_reference_and_stays_in_sublevel_set(self):
        X = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
        y = np.array([1., -1., 1., -1.])
        penalty = 0.1
        history, final = run_regularized(X, y, np.array([1., 0.]), penalty,
                                         n_steps=1000, tolerance=1e-9)
        self.assertEqual(final["stop_reason"], "gradient_tolerance")
        self.assertLess(final["distance_to_reference"], 1e-7)
        self.assertLessEqual(final["objective_gap"], final["objective_gap_bound"] + 1e-14)
        objectives = [row["penalized_objective"] for row in history]
        self.assertTrue(np.all(np.diff(objectives) <= 1e-14))
        self.assertTrue(all(row["weight_norm"] <= np.sqrt(2 * np.log(2) / penalty)
                            for row in history))
        reference = regularized_reference(X, y, penalty)
        self.assertLess(np.linalg.norm(regularized_gradient(X, y, reference, penalty)), 1e-12)

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
