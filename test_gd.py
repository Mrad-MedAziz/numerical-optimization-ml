import numpy as np  # type: ignore
from optimization.linear_model import LinearRegression # type: ignore

# 1. Generate synthetic linear data with known ground truth
np.random.seed(42)
X = 2 * np.random.rand(100, 1) #We build a Matrix of size 100*1 instead of a simple vector because the functions we use to create A_b later
#require a Matrix instead of a vector

# True intercept = 4.0, True slope = 3.0, small Gaussian noise
y = 4.0 + 3.0 * X.squeeze() + np.random.randn(100) * 0.05

# 2. Fit model using batch gradient descent
model = LinearRegression()
theta = model.fit_gradient_descent(X, y, lr=0.1, epochs=1000)

# 3. Inspect convergence results
print(f"Estimated intercept (theta_0): {theta[0]:.4f} (Expected: ~4.0000)")
print(f"Estimated slope     (theta_1): {theta[1]:.4f} (Expected: ~3.0000)")
print(f"Final cost          (MSE / 2): {model.cost:.6f}")