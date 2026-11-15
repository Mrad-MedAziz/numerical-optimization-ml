import os
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from optimization import LogisticRegression


def run_benchmark():
  # 1. Generate synthetic dataset with controlled overlap
  np.random.seed(42)
  n_per_class = 150
  X_0 = np.random.randn(n_per_class, 2) + np.array([-1.0, -1.0])
  X_1 = np.random.randn(n_per_class, 2) + np.array([1.0, 1.0])

  X = np.vstack([X_0, X_1])
  y = np.array([0.0] * n_per_class + [1.0] * n_per_class)

  # 2. Fit first-order method: gradient ascent
  model_ga = LogisticRegression()
  model_ga.fit_gradient_ascent(X, y, lr=0.5, epochs=250)

  # 3. Fit second-order method: Newton-Raphson
  model_newton = LogisticRegression()
  model_newton.fit_newton(X, y, max_iter=20, tol=1e-8)

  # 4. Reference optimum ell* from a separate, tighter Newton run.
  #    This keeps Newton's own final gap from being zero by construction.
  ref = LogisticRegression()
  ref.fit_newton(X, y, max_iter=60, tol=1e-12)
  ell_star = ref.log_likelihood_history[-1]

  # 5. Optimality gap: |ell(theta_t) - ell*|
  history_newton = np.array(model_newton.log_likelihood_history)
  history_ga = np.array(model_ga.log_likelihood_history)

  gap_ga = np.abs(history_ga - ell_star)
  gap_newton = np.abs(history_newton - ell_star)

  # Clip exact zeros and values below float precision for the log-scale plot
  eps = 1e-16
  gap_ga = np.clip(gap_ga, eps, None)
  gap_newton = np.clip(gap_newton, eps, None)

  # 6. Plot convergence
  os.makedirs("figures", exist_ok=True)
  plt.figure(figsize=(9, 5.5), dpi=150)

  n_ga_updates = len(gap_ga) - 1
  n_newton_updates = len(gap_newton) - 1

  plt.plot(
      range(len(gap_ga)),
      gap_ga,
      label=f"Gradient Ascent (lr=0.5, {n_ga_updates} updates)",
      color="#1f77b4",
      linewidth=2,
  )
  plt.plot(
      range(len(gap_newton)),
      gap_newton,
      label=f"Newton-Raphson / IRLS ({n_newton_updates} updates)",
      color="#d62728",
      linewidth=2.5,
      marker="o",
      markersize=5,
  )

  plt.yscale("log")
  plt.xlim(0, 35)  # Truncated to t=35 to show the quadratic phase
  plt.xlabel("Optimization Step ($t$)", fontsize=11)
  plt.ylabel(
      r"Optimality Gap $|\ell(\theta^{(t)}) - \ell(\theta^*)|$ (Log Scale)",
      fontsize=11,
  )
  plt.title(
      "Convergence Rate: First-Order vs. Second-Order Optimization",
      fontsize=12,
      fontweight="bold",
  )
  plt.grid(True, which="both", linestyle="--", alpha=0.4)
  plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
  plt.tight_layout()

  output_path = "figures/convergence.png"
  plt.savefig(output_path)
  plt.close()

  print("[+] Benchmark completed successfully.")
  print(f"    - Newton-Raphson updates: {n_newton_updates}")
  print(f"    - Gradient Ascent updates: {n_ga_updates}")
  print(f"    - Convergence plot saved to:  {output_path}")


if __name__ == "__main__":
  run_benchmark()