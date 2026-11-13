import numpy as np # type: ignore
from optimization.base import add_bias


class LogisticRegression:

  def __init__(self):
    self.theta = None
    self.log_likelihood_history = []
    self.likelihood = 0.0

  @staticmethod  #is a built-in decorator that defines a method inside a class that does not depend on or modify the instance (self)
  def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_safe = np.clip(z, -500.0, 500.0)  #This prevents np.exp(-z) from exceeding IEEE 754 float limits and producing inf or overflow warnings.
    return 1.0 / (1.0 + np.exp(-z_safe))

  def _cost(self, y: np.ndarray, p: np.ndarray) -> float:
    n = p.shape[0]
    eps = 1e-15
    p_safe = np.clip(p, eps, 1.0 - eps) #to avoid having log(0.0) which generates an error
    # Average log-likelihood: values increase towards 0.0
    return (1.0 / n) * (y @ np.log(p_safe) + (1.0 - y) @ np.log(1.0 - p_safe))

  def fit_gradient_ascent(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000) -> np.ndarray:
    n, d = X.shape
    X_b = add_bias(X)
    self.theta = np.zeros(d + 1)
    self.log_likelihood_history = []

    for _ in range(epochs):
      z = X_b @ self.theta
      p = self._sigmoid(z)

      ll = self._cost(y, p)
      self.log_likelihood_history.append(ll)

      grad = (1.0 / n) * (X_b.T @ (y - p))

      self.theta += lr * grad

    self.likelihood = np.exp(self.log_likelihood_history[-1])
    return self.theta

  def fit_newton (self, X: np.ndarray, y: np.ndarray, max_iter : int = 20, tol: float  = 1e-6,eps_reg : float = 1e-8 ) -> np.ndarray:
    n,d = X.shape
    X_b = add_bias(X)
    self.theta = np.zeros(d + 1)
    self.log_likelihood_history = []

    for _ in range (max_iter) :
      z = X_b @ self.theta
      p = self._sigmoid(z)

      ll = self._cost(y, p)
      self.log_likelihood_history.append(ll)

      grad = (1.0 / n) * (X_b.T @ (y - p))

      w = (p * (1.0 - p)).reshape(n, 1)  # shape (n, 1)
      H = -(1.0 / n) * (X_b.T @ (w * X_b))  # shape (d+1, d+1)
      # If probabilities saturate near 0.0 or 1.0, w_i = p_i(1 - p_i) tends to 0, which can make the Hessian singular or ill-conditioned. 
      # Adding a tiny perturbation (10^{-8}) guarantees negative definiteness and invertibility.

      # Numerical damping along diagonal ensures invertibility
      H_reg = H - eps_reg * np.eye(d + 1) #np.eye(N) creates an N * N square identity matrix.

      delta = np.linalg.solve(H_reg, grad)
      if np.linalg.norm(delta) < tol:
        break
      self.theta -= delta

    self.likelihood = np.exp(self.log_likelihood_history[-1])
    return self.theta


  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    X_b = add_bias(X)
    return self._sigmoid(X_b @ self.theta)

  def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (self.predict_proba(X) >= threshold).astype(float)
  