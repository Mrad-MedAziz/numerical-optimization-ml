import numpy as np # type: ignore
from optimization.logistic_model import LogisticRegression

# Synthetic 2D classification dataset with overlap
np.random.seed(42)
X_0 = np.random.randn(100, 2) + np.array([-1.0, -1.0])
X_1 = np.random.randn(100, 2) + np.array([1.0, 1.0])
X = np.vstack([X_0, X_1])
y = np.array([0.0] * 100 + [1.0] * 100)

# 1. Newton-Raphson
model_newton = LogisticRegression()
model_newton.fit_newton(X, y)

# 2. Gradient Ascent
model_ga = LogisticRegression()
model_ga.fit_gradient_ascent(X, y, lr=0.5, epochs=500)

print(f"Newton-Raphson converged in {len(model_newton.log_likelihood_history)} iterations")
print(f"Newton Final Likelihood: {model_newton.likelihood:.6f}")
print(f"Gradient Ascent Final Likelihood: {model_ga.likelihood:.6f}")