from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from optimization.separable_logistic import logistic_loss, logistic_gradient, sigmoid, gradient_descent, max_margin_solution

OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "separable"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    X = np.random.randn(100,2)
    X = X[ X[:,0] != 0]

    y = np.where((X[:,0]) > 0 , 1 , -1 )

    w_separator = np.array([1.0,0.0])

    signed_scores = y * (X @ w_separator)

    assert np.all(signed_scores > 0), "Data is not strictly separated"
    print("Shape:", X.shape)
    print("Classes and counts:", np.unique(y, return_counts=True))
    print(f"The functional margin is {np.min(signed_scores)}")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
    plt.axvline(0)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.axis("equal")
    plt.savefig(OUTPUT / "separable_data.png", dpi=150, bbox_inches="tight")

    run_experiments(X, y)

    w_hat, multipliers = max_margin_solution(X, y)
    svm_scores = y * (X @ w_hat)
    support = np.isclose(svm_scores, 1.0, atol=1e-6, rtol=0)
    svm_direction = w_hat / np.linalg.norm(w_hat)

    w, losses, gradient_norms, weight_norms, directions, direction_errors, margins = gradient_descent(
        X, y, w_hat, print_every=10000)

    s = y * (X @ w)
    coefficients = -y * sigmoid (-s)
    contributions_to_gradient = (coefficients.reshape(-1, 1) * X) / X.shape[0]
    contribution_norms = np.linalg.norm(contributions_to_gradient, axis=1)
    largest = np.argsort(contribution_norms)[-5:][::-1]
    support_gradient = contributions_to_gradient[support].sum(axis=0)
    other_gradient = contributions_to_gradient[~support].sum(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    axes[0].plot(losses)
    axes[0].set_title("Loss")

    axes[1].plot(gradient_norms)
    axes[1].set_title("Gradient norm")

    axes[2].plot(weight_norms)
    axes[2].set_title("Weight norm")

    directions = np.array(directions)
    updates = np.arange(1, len(directions) + 1)

    axes[3].plot(updates, directions[:, 0])
    axes[3].set_title("Direction: first component")
    axes[3].set_xlabel("Updates")
    axes[3].axhline(svm_direction[0], color="black", linestyle="--", label="SVM")
    axes[3].legend()

    axes[4].plot(updates, directions[:, 1])
    axes[4].set_title("Direction: second component")
    axes[4].set_xlabel("Updates")
    axes[4].axhline(svm_direction[1], color="black", linestyle="--", label="SVM")
    axes[4].legend()

    axes[5].plot(updates, direction_errors)
    axes[5].set_title("Distance to SVM direction")
    axes[5].set_xlabel("Updates")
    axes[5].set_ylabel("Euclidean distance")

    fig.tight_layout()
    fig.savefig(OUTPUT / "gradient_descent_curves.png", dpi=150, bbox_inches="tight")

    svm_margin = 1 / np.linalg.norm(w_hat)
    margin_fig, margin_axis = plt.subplots(figsize=(8, 4))
    margin_axis.plot(updates, margins, label="Gradient descent")
    margin_axis.axhline(svm_margin, color="black", linestyle="--", label="SVM maximum margin")
    margin_axis.set_xlabel("Updates")
    margin_axis.set_ylabel("Minimum signed geometric margin")
    margin_axis.set_title("Geometric margin during gradient descent")
    margin_axis.legend()
    margin_fig.tight_layout()
    margin_fig.savefig(OUTPUT / "geometric_margin.png", dpi=150, bbox_inches="tight")
    
    print("Final geometric margin:", margins[-1])
    print("SVM maximum margin:", svm_margin)

    print("Support-vector indices:", np.where(support)[0])
    print("Support vectors:", X[support])
    print("Their labels:", y[support])
    print("SVM weights:", w_hat)
    print("SVM direction:", svm_direction)
    
    check_support_vectors(svm_scores, multipliers)
    plot_separators(X, y, w, w_hat, support)
    
    for i in largest:
        print("Point:", i)
        print("Contribution norm:", contribution_norms[i])
        print("SVM support vector:", support[i])

    print("Support gradient:", support_gradient)
    print("Non-support gradient:", other_gradient)
    print("Total gradient:", contributions_to_gradient.sum(axis=0))
    print("Current directional error:", np.linalg.norm(w / np.linalg.norm(w) - svm_direction))
    for label, gradient in [("Support only", support_gradient),
                                    ("Other points only", other_gradient),
                                    ("Combined", support_gradient + other_gradient)]:
        trial_w = w - 0.1 * gradient
        trial_error = np.linalg.norm(trial_w / np.linalg.norm(trial_w) - svm_direction)
        print(label, "directional error after one hypothetical step:", trial_error)
    
    plt.show()


def check_support_vectors(scores, multipliers):
    # Check whether our support-vector selection depends on the tolerance.
    print("\nSupport-vector tolerance check")
    for tolerance in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        selected = np.isclose(scores, 1.0, atol=tolerance, rtol=0)
        print("Tolerance:", tolerance, "Indices:", np.where(selected)[0])

    print("\nScores and solver multipliers for the original selection")
    selected = np.isclose(scores, 1.0, atol=1e-6, rtol=0)
    for i in np.where(selected)[0]:
        print("Point:", i)
        print(f"Score: {scores[i]:.17g}")
        print(f"Score minus 1: {scores[i] - 1:.17g}")
        print(f"Solver multiplier: {multipliers[i]:.12g}")

    other_scores = scores[~selected]
    if len(other_scores) > 0:
        print("Smallest score outside that selection:", np.min(other_scores))


def plot_separators(X, y, w_final, w_hat, support):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    heights = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 300)
    support_indices = np.where(support)[0]
    colors = ["#d88b00", "#9a35b5"]

    for ax in axes:
        ax.scatter(X[y == -1, 0], X[y == -1, 1], color="#3578bb",
                    s=25, alpha=0.6, label="Class -1")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#ce5547",
                    s=25, alpha=0.6, label="Class +1")
        ax.axvline(0, color="#777777", linestyle=":", linewidth=2,
                    label="Original separator: x1 = 0")
        # Solve w[0]*x1 + w[1]*x2 = 0 for x1.
        ax.plot(-w_final[1] * heights / w_final[0], heights,
                color="#148359", linewidth=2, label="GD separator: 100,000 updates")
        ax.plot(-w_hat[1] * heights / w_hat[0], heights,
                color="black", linestyle="--", linewidth=2, label="SVM separator")
        ax.plot((1 - w_hat[1] * heights) / w_hat[0], heights,
                color="black", linestyle="-.", alpha=0.45,
                label="Positive SVM margin: score = 1")
        ax.plot(0, 0, "+", color="black", markersize=9)
        for number, index in enumerate(support_indices):
            point = X[index]
            color = colors[number % len(colors)]
            ax.annotate("", xy=point, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.8))
            ax.scatter(point[0], point[1], s=150, color=color, edgecolor="black",
                        zorder=6, label=f"Support vector {index} (label +1)")
        ax.set_xlabel("First feature x1")
        ax.set_ylabel("Second feature x2")
        ax.grid(alpha=0.15)

    axes[0].set_title("Whole dataset — equal axis scale")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
    axes[0].set_ylim(heights.min(), heights.max())

    axes[1].set_title("Zoom near the origin — horizontal scale expanded")
    axes[1].set_xlim(-0.025, 0.03)
    axes[1].set_ylim(-0.5, 1.75)
    for number, index in enumerate(support_indices):
        point = X[index]
        axes[1].annotate(f"Point {index}\n({point[0]:.5f}, {point[1]:.3f})",
                            xy=point, xytext=(12, 12), textcoords="offset points",
                            color=colors[number % len(colors)], fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Two support vectors and three separating lines", fontsize=15)
    fig.tight_layout(rect=(0, 0.23, 1, 0.95))
    fig.savefig(OUTPUT / "separators_support_vectors.png", dpi=180, bbox_inches="tight")


def run_experiments(X, y):
    weight_vectors = [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                np.array([2.0, 0.0]), np.array([4.0, 0.0]),
                                np.array([8.0, 0.0]), np.array([16.0, 0.0])]
    for w in weight_vectors:
        print("Weights:", w)
        print("Loss:", logistic_loss(X, y, w))
        print("Gradient:", logistic_gradient(X, y, w))
        print()

if __name__ == "__main__":
    main()
