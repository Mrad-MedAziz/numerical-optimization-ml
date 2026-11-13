import numpy as np # type: ignore
from optimization.linear_model import LinearRegression

np.random.seed(42)
X = 2.0 * np.random.rand(100, 1)
y = 4.0 + 3.0 * X.squeeze() + 0.05 * np.random.randn(100)

# Model 1: Analytical Normal Equations
model_exact = LinearRegression()
theta_exact = model_exact.fit_normal_equations(X, y)

# Model 2: Batch Gradient Descent (Fixed Step Size)
model_fixed = LinearRegression()
theta_fixed = model_fixed.fit_gradient_descent(X, y, lr = 0.05)

# Model 3: Batch Gradient Descent (Armijo Backtracking Line Search)
model_armijo = LinearRegression()
theta_armijo = model_armijo.fit_armijo_gradient_descent(X, y)

# Print Summary Results
print("=" * 80)
print(f"{'Method':<25} | {'theta_0 (Intercept)':<12} | {'theta_1 (Slope)':<10} | {'Final Cost':<10}")
print("=" * 80)
print(f"{'Normal Equations':<25} | {theta_exact[0]:<19.4f} | {theta_exact[1]:<15.4f} | {model_exact.cost:<10.6f}")
print(f"{'GD (Fixed lr=0.05)':<25} | {theta_fixed[0]:<19.4f} | {theta_fixed[1]:<15.4f} | {model_fixed.cost:<10.6f}")
print(f"{'GD (Armijo Line Search)':<25} | {theta_armijo[0]:<19.4f} | {theta_armijo[1]:<15.4f} | {model_armijo.cost:<10.6f}")
print("=" * 80)

# Convergence Verification Checks
diff_fixed = np.linalg.norm(theta_exact - theta_fixed)
diff_armijo = np.linalg.norm(theta_exact - theta_armijo)

print(f"L2 error (Fixed vs. Exact):   {diff_fixed:.6f}")
print(f"L2 error (Armijo vs. Exact):  {diff_armijo:.6f}")

assert diff_fixed < 0.05, f"Fixed GD diverged: L2 error {diff_fixed:.6f} >= 0.05"
assert diff_armijo < 0.05, f"Armijo GD diverged: L2 error {diff_armijo:.6f} >= 0.05"

print("\nAll solver assertions passed successfully!")