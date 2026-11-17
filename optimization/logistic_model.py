import numpy as np # type: ignore
from optimization.base import add_bias


class LogisticRegression:

    def __init__(self):
        self.theta = None
        self.log_likelihood_history = []
        self.likelihood = 0.0
        self.n_iter_ = 0
        self.converged_ = False

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Evaluate the sigmoid without overflowing the exponential."""
        exp_neg_abs = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + exp_neg_abs),
                        exp_neg_abs / (1.0 + exp_neg_abs))

    @staticmethod
    def _log_likelihood(y: np.ndarray, z: np.ndarray) -> float:
        """Mean Bernoulli log-likelihood, evaluated directly from logits."""
        signed_z = np.where(y == 1.0, -z, z)
        return float(-np.mean(np.logaddexp(0.0, signed_z)))

    @staticmethod
    def _validate_data(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be a nonempty 2D matrix and y a matching 1D vector")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("Logistic regression requires binary labels 0 or 1")
        return X, y

    def _cost(self, y: np.ndarray, p: np.ndarray) -> float:
        """Compatibility helper for probabilities; fitting uses the stable logit form."""
        n = p.shape[0]
        eps = 1e-15
        p_safe = np.clip(p, eps, 1.0 - eps) #to avoid having log(0.0) which generates an error
        # Average log-likelihood: values increase towards 0.0
        return (1.0 / n) * (y @ np.log(p_safe) + (1.0 - y) @ np.log(1.0 - p_safe))

    def fit_gradient_ascent(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000, tol: float = 1e-5) -> np.ndarray:
        X, y = self._validate_data(X, y)
        n, d = X.shape
        X_b = add_bias(X)
        self.theta = np.zeros(d + 1)
        self.log_likelihood_history = []

        if epochs < 0 or int(epochs) != epochs or lr <= 0 or tol < 0:
            raise ValueError("epochs must be a nonnegative integer; lr > 0 and tol >= 0")
        self.n_iter_ = 0
        self.converged_ = False
        for step in range(epochs + 1):
            z = X_b @ self.theta
            p = self._sigmoid(z)

            ll = self._log_likelihood(y, z)
            self.log_likelihood_history.append(ll)

            grad = (1.0 / n) * (X_b.T @ (y - p))

            if np.linalg.norm(grad) < tol:
                self.converged_ = True
                break

            if step == epochs:
                break
            self.theta += lr * grad
            self.n_iter_ += 1

        self.likelihood = np.exp(self.log_likelihood_history[-1])
        return self.theta

    def fit_newton(self, X: np.ndarray, y: np.ndarray, max_iter: int = 20, tol: float = 1e-5, eps_reg: float = 1e-8) -> np.ndarray:
        X, y = self._validate_data(X, y)
        n, d = X.shape
        X_b = add_bias(X)
        self.theta = np.zeros(d + 1)
        self.log_likelihood_history = []

        if max_iter < 0 or int(max_iter) != max_iter or tol < 0 or eps_reg < 0:
            raise ValueError("max_iter must be a nonnegative integer; tol and eps_reg >= 0")
        self.n_iter_ = 0
        self.converged_ = False
        for step in range(max_iter + 1):
            z = X_b @ self.theta
            p = self._sigmoid(z)

            ll = self._log_likelihood(y, z)
            self.log_likelihood_history.append(ll)

            grad = (1.0 / n) * (X_b.T @ (y - p))
            if np.linalg.norm(grad) < tol:
                self.converged_ = True
                break

            if step == max_iter:
                break

            w = (p * (1.0 - p)).reshape(n, 1)  # shape (n, 1)
            H = -(1.0 / n) * (X_b.T @ (w * X_b))  # shape (d+1, d+1)
            # If probabilities saturate near 0.0 or 1.0, w_i = p_i(1 - p_i) tends to 0, which can make the Hessian singular or ill-conditioned. 
            # Positive damping makes the curvature matrix negative definite in exact arithmetic.

            # Numerical damping along diagonal ensures invertibility
            H_reg = H - eps_reg * np.eye(d + 1) #np.eye(N) creates an N * N square identity matrix.

            delta = np.linalg.solve(H_reg, grad)
            self.theta -= delta
            self.n_iter_ += 1

        self.likelihood = np.exp(self.log_likelihood_history[-1])
        return self.theta


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_b = add_bias(X)
        return self._sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(float)