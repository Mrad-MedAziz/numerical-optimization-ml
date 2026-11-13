import numpy as np # type: ignore
from optimization.logistic_model import LogisticRegression 

# Synthetic linearly separable clusters
np.random.seed(42)
X_0 = np.random.randn(50, 2) - 2.0
X_1 = np.random.randn(50, 2) + 2.0
X = np.vstack([X_0, X_1])
y = np.array([0.0] * 50 + [1.0] * 50)

model = LogisticRegression()
model.fit_gradient_ascent(X, y, lr=0.5, epochs=300)

preds = model.predict(X)
acc = np.mean(preds == y)

print(f"Initial Log-Likelihood: {model.log_likelihood_history[0]:.4f}")
print(f"Final Log-Likelihood:   {model.log_likelihood_history[-1]:.4f}")
print(f"Accuracy:               {acc * 100:.1f}%")