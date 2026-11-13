import numpy as np  # type: ignore
from optimization.base import add_bias, backtracking_line_search  # type: ignore


class LinearRegression:

  def __init__(self):
    self.theta = None
    self.loss_history = []
    self.cost = 0.0

  def fit_normal_equations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve Ordinary Least Squares analytically via the Normal Equations:

    (X_b^T * X_b) * theta = X_b^T * y
    """
    n = X.shape[0]
    X_b = add_bias(X)

    A = X_b.T @ X_b
    b = X_b.T @ y

    # Direct solve avoids explicitly computing the matrix inverse
    self.theta = np.linalg.solve(A, b)

    residuals = (X_b @ self.theta) - y
    self.cost = (1.0 / (2.0 * n)) * (residuals @ residuals)
    return self.theta

  def fit_gradient_descent(
      self,
      X: np.ndarray,
      y: np.ndarray,
      lr: float = 0.05,
      epochs: int = 1000,
  ) -> np.ndarray:
    """Baseline solver: First-order batch gradient descent with a fixed step size."""
    n, d = X.shape
    X_b = add_bias(X)
    self.theta = np.zeros(d + 1)
    self.loss_history = []

    for _ in range(epochs):
      residuals = (X_b @ self.theta) - y
      cost = (1.0 / (2.0 * n)) * (residuals @ residuals)
      self.loss_history.append(cost)

      grad = (1.0 / n) * (X_b.T @ residuals)
      self.theta -= lr * grad

    self.cost = self.loss_history[-1]
    return self.theta

  def fit_armijo_gradient_descent(
      self,
      X: np.ndarray,
      y: np.ndarray,
      epochs: int = 1000,
  ) -> np.ndarray:
    """Enhanced solver: Gradient descent with Armijo backtracking line search

    to adaptively choose step size alpha and guarantee sufficient decrease.
    """
    n, d = X.shape
    X_b = add_bias(X)
    self.theta = np.zeros(d + 1)
    self.loss_history = []

    cost_fn = lambda t: (1.0 / (2.0 * n)) * np.sum(((X_b @ t) - y) ** 2)

    for _ in range(epochs):
      residuals = (X_b @ self.theta) - y
      cost = (1.0 / (2.0 * n)) * (residuals @ residuals)
      self.loss_history.append(cost)

      grad = (1.0 / n) * (X_b.T @ residuals)
      alpha = backtracking_line_search(cost_fn, self.theta, grad, alpha=1.0)
      self.theta -= alpha * grad

    self.cost = self.loss_history[-1]
    return self.theta

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Compute linear predictions for feature matrix X."""
    X_b = add_bias(X)
    return X_b @ self.theta