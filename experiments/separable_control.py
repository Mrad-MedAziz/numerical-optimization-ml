from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from optimization.separable_logistic import max_margin_solution, gradient_descent


def main():
    X = np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]])
    y = np.array([1., -1., 1., -1.])
    w_exact = np.array([1., 0.])

    print("Signed scores at the exact solution:", y * (X @ w_exact))
    w_solver, multipliers = max_margin_solution(X, y)
    print("Gurobi solution:", w_solver)
    if not np.allclose(w_solver, w_exact, atol=1e-6, rtol=0):
        raise RuntimeError("Gurobi disagrees with the known solution")

    # Use the exact direction to avoid numerical reference error in this case.
    w, losses, gradient_norms, weight_norms, directions, errors, margins = gradient_descent(
        X, y, w_exact, lr=0.1, n_steps=100000, print_every=10000)
    updates = np.arange(1, len(errors) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(losses)
    axes[0, 0].set_title("Mean logistic loss")
    axes[0, 1].plot(weight_norms)
    axes[0, 1].set_title("Weight norm")
    axes[1, 0].plot(updates, errors)
    axes[1, 0].set_title("Distance to exact SVM direction")
    axes[1, 1].plot(updates, margins, label="Gradient descent")
    axes[1, 1].axhline(1, color="black", linestyle="--", label="Maximum margin = 1")
    axes[1, 1].set_title("Minimum signed geometric margin")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Updates")
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / "figures" / "separable"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "four_points_curves.png", dpi=150, bbox_inches="tight")

    summary = ["Four-point experiment: lr=0.1, zero initialization, no intercept",
               "Exact SVM solution: (1, 0); maximum margin: 1",
               f"Gurobi solution: {w_solver}"]
    for step in [1, 1000, 10000, 100000]:
        summary.append(f"Update {step}: loss={losses[step]:.10f}, "
                       f"weight norm={weight_norms[step]:.8f}, "
                       f"direction error={errors[step - 1]:.8f}, "
                       f"margin={margins[step - 1]:.8f}")
    summary.append(f"Final weights: {w}")
    text = "\n".join(summary)
    print("\n" + text)
    (Path(__file__).resolve().parents[1] / "results" / "separable").mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parents[1] / "results" / "separable" / "four_points_results.txt").write_text(text + "\n")
    plt.show()


if __name__ == "__main__":
    main()
