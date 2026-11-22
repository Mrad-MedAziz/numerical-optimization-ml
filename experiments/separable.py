from pathlib import Path
import contextlib
import csv
import io
import json
import time
import sys
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from optimization.separable_logistic import logistic_loss, logistic_gradient, max_margin_solution

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "separable"
FIGURES = ROOT / "figures" / "separable"
STEPS = 100000
TOLERANCES = [1e-2, 1e-3, 1e-4, 1e-5]
FRACTIONS = [0.1, 0.5, 1.0]
SEEDS = [42, 0, 1]


def datasets():
    yield "four_points", -1, np.array([[1., 0.], [-1., 0.], [2., 1.], [-2., -1.]]), np.array([1., -1., 1., -1.])
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        X = rng.randn(100, 2)
        X = X[X[:, 0] != 0]
        y = np.where(X[:, 0] > 0, 1., -1.)
        yield "gaussian_sign", seed, X, y
        # Same observations/labels, with additional clearance from x1=0.
        X_gap = X.copy()
        X_gap[:, 0] = y * (np.abs(X[:, 0]) + 0.3)
        yield "gaussian_gap", seed, X_gap, y


def save_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_dataset(X, y):
    assert X.shape == (len(y), 2)
    assert set(y) == {-1., 1.}
    assert np.all(y * X[:, 0] > 0)
    w = np.array([0.3, -0.2])
    h = 1e-5
    numerical = np.zeros(2)
    for j in range(2):
        change = np.zeros(2)
        change[j] = h
        numerical[j] = (logistic_loss(X, y, w + change) - logistic_loss(X, y, w - change)) / (2 * h)
    analytic = logistic_gradient(X, y, w)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
    return np.max(np.abs(analytic - numerical))


def reference_solution(X, y):
    with contextlib.redirect_stdout(io.StringIO()):
        w_hat, dual = max_margin_solution(X, y)
    A = y[:, None] * X
    slack = A @ w_hat - 1
    objective = 0.5 * (w_hat @ w_hat)
    dual_objective = dual.sum() - 0.5 * np.linalg.norm(A.T @ dual)**2
    checks = {
        "primal_violation": max(0., -slack.min()),
        "dual_violation": max(0., -dual.min()),
        "stationarity_residual": np.max(np.abs(w_hat - A.T @ dual)),
        "complementarity_residual": np.max(np.abs(dual * slack)),
        "relative_objective_gap": abs(objective - dual_objective) / max(1., abs(objective)),
    }
    assert checks["primal_violation"] < 1e-6
    assert checks["dual_violation"] < 1e-6
    assert checks["stationarity_residual"] / max(1., np.linalg.norm(w_hat)) < 1e-6
    assert checks["complementarity_residual"] / max(1., objective) < 1e-6
    assert checks["relative_objective_gap"] < 1e-6
    return w_hat, checks


def measurements(X, y, w, gradient, target, maximum_margin):
    norm = np.linalg.norm(w)
    signed = y * (X @ w)
    # Direction and geometric margin are undefined for zero weights.
    error = np.linalg.norm(w / norm - target) if norm > 0 else float("nan")
    margin = signed.min() / norm if norm > 0 else float("nan")
    return {
        "loss": float(logistic_loss(X, y, w)),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "weight_norm": float(norm),
        "direction_error": float(error),
        "geometric_margin": float(margin),
        "margin_ratio": float(margin / maximum_margin),
        "minimum_signed_score": float(signed.min()),
        "strictly_correct_fraction": float(np.mean(signed > 0)),
    }


def run_one(X, y, w_hat, lr, metadata, n_steps=STEPS, tolerances=None):
    if tolerances is None:
        tolerances = TOLERANCES
    if n_steps < 0 or int(n_steps) != n_steps or not np.isfinite(lr) or lr <= 0:
        raise ValueError("n_steps must be a nonnegative integer and lr finite and positive")
    if any(not np.isfinite(tol) or tol < 0 for tol in tolerances):
        raise ValueError("Tolerances must be finite and nonnegative")
    w = np.zeros(X.shape[1])
    target = w_hat / np.linalg.norm(w_hat)
    maximum_margin = 1 / np.linalg.norm(w_hat)
    checkpoints = {0, n_steps}
    if n_steps > 0:
        checkpoints |= set(np.geomspace(1, n_steps, 150).astype(int)) | {1000, 10000}
    snapshots, crossings = [], {}
    gradient = logistic_gradient(X, y, w)
    previous_loss = logistic_loss(X, y, w)
    increases = 0
    final = None
    for step in range(n_steps + 1):
        current_loss = logistic_loss(X, y, w)
        if current_loss > previous_loss + 1e-12:
            increases += 1
        previous_loss = current_loss
        new_thresholds = [tol for tol in tolerances if tol not in crossings and np.linalg.norm(gradient) < tol]
        if step in checkpoints or new_thresholds:
            measured = measurements(X, y, w, gradient, target, maximum_margin)
            row = dict(metadata, update=step, **measured)
            if step in checkpoints:
                snapshots.append(row)
            for tol in new_thresholds:
                crossings[tol] = dict(row, tolerance=tol, stop_reason="gradient_tolerance")
            if step == n_steps:
                final = row
        if step == n_steps:
            break
        w = w - lr * gradient
        gradient = logistic_gradient(X, y, w)
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("Nonfinite iterate")
    for tol in tolerances:
        if tol not in crossings:
            crossings[tol] = dict(final, tolerance=tol, stop_reason="iteration_budget")
    assert increases == 0, "Loss increased despite the selected safe step"
    return snapshots, list(crossings.values()), dict(final, loss_increases=increases)


def plot_results(history, stopping):
    labels = {"four_points": "Four-point dataset", "gaussian_sign": "Original random dataset", "gaussian_gap": "Random dataset with clearance"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for family in ["four_points", "gaussian_sign", "gaussian_gap"]:
        seed = -1 if family == "four_points" else 42
        rows = [r for r in history if r["dataset"] == family and r["seed"] == seed and r["step_fraction"] == 0.5 and r["update"] > 0]
        steps = [r["update"] for r in rows]
        axes[0, 0].loglog(steps, [r["loss"] for r in rows], label=labels[family])
        axes[0, 1].semilogx(steps, [r["direction_error"] for r in rows], label=labels[family])
        axes[1, 0].semilogx(steps, [r["margin_ratio"] for r in rows], label=labels[family])
        axes[1, 1].semilogx(steps, [r["weight_norm"] for r in rows], label=labels[family])
    titles = ["Mean loss", "Distance to SVM direction", "Geometric margin / maximum margin", "Weight norm"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Updates (log scale)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[1, 0].axhline(1, color="black", linestyle="--", linewidth=1)
    fig.suptitle("Same step fraction 0.5/B; random datasets use seed 42")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, family in zip(axes, ["four_points", "gaussian_sign", "gaussian_gap"]):
        seed = -1 if family == "four_points" else 42
        for fraction in FRACTIONS:
            rows = [r for r in history if r["dataset"] == family and r["seed"] == seed and r["step_fraction"] == fraction and r["update"] > 0]
            ax.semilogx([r["update"] for r in rows], [r["direction_error"] for r in rows], label=f"lr={fraction}/B")
        ax.set_title(labels[family])
        ax.set_xlabel("Updates (log scale)")
        ax.set_ylabel("Directional error")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "step_size_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, family in zip(axes, ["gaussian_sign", "gaussian_gap"]):
        for seed in SEEDS:
            rows = [r for r in history if r["dataset"] == family and r["seed"] == seed and r["step_fraction"] == 0.5 and r["update"] > 0]
            ax.semilogx([r["update"] for r in rows], [r["direction_error"] for r in rows], label=f"Seed {seed}")
        ax.set_title(labels[family])
        ax.set_xlabel("Updates (log scale)")
        ax.set_ylabel("Directional error")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Seed comparison at learning rate 0.5/B")
    fig.tight_layout()
    fig.savefig(FIGURES / "seed_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family in ["four_points", "gaussian_sign", "gaussian_gap"]:
        seed = -1 if family == "four_points" else 42
        rows = [r for r in stopping if r["dataset"] == family and r["seed"] == seed and r["step_fraction"] == 0.5]
        for ax, metric in zip(axes, ["direction_error", "weight_norm"]):
            line, = ax.plot([r["tolerance"] for r in rows], [r[metric] for r in rows], label=labels[family])
            for r in rows:
                marker = "o" if r["stop_reason"] == "gradient_tolerance" else "x"
                ax.scatter(r["tolerance"], r[metric], marker=marker, color=line.get_color(), zorder=3)
    for ax, title in zip(axes, ["Directional error when stopping", "Weight norm when stopping"]):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Gradient tolerance (tighter to the right)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Circle: threshold reached; cross: budget reached instead. lr=0.5/B")
    fig.tight_layout()
    fig.savefig(FIGURES / "stopping_comparison.png", dpi=160)
    plt.close(fig)


def main():
    import gurobipy as gp
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_history, all_stops, all_final, references = [], [], [], []
    start = time.monotonic()
    for family, seed, X, y in datasets():
        gradient_error = verify_dataset(X, y)
        w_hat, checks = reference_solution(X, y)
        if family == "four_points":
            np.testing.assert_allclose(w_hat, [1., 0.], atol=1e-6, rtol=0)
        B = np.linalg.norm(X, ord=2)**2 / (4 * len(y))
        references.append(dict(dataset=family, seed=seed, n=len(y), B=B,
                               w1=w_hat[0], w2=w_hat[1], maximum_margin=1 / np.linalg.norm(w_hat),
                               gradient_check_error=gradient_error, **checks))
        np.savez(OUTPUT / f"data_{family}_{seed}.npz", X=X, y=y, w_hat=w_hat)
        for fraction in FRACTIONS:
            lr = fraction / B
            assert 0 < lr < 2 / B
            metadata = dict(dataset=family, seed=seed, step_fraction=fraction, lr=lr)
            history, stops, final = run_one(X, y, w_hat, lr, metadata)
            all_history.extend(history)
            all_stops.extend(stops)
            all_final.append(final)
            print(f"Finished {family}, seed={seed}, lr={lr:.5g}", flush=True)
            # Save completed runs incrementally.
            save_csv("trajectory_samples.csv", all_history)
            save_csv("stopping_comparison.csv", all_stops)
            save_csv("final_results.csv", all_final)
        save_csv("reference_checks.csv", references)
    plot_results(all_history, all_stops)
    config = dict(steps=STEPS, tolerances=TOLERANCES, step_fractions=FRACTIONS,
                  seeds=SEEDS, numpy=np.__version__, python=sys.version,
                  matplotlib=matplotlib.__version__, gurobi=list(gp.gurobi.version()),
                  elapsed_seconds=time.monotonic() - start)
    (OUTPUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print("All 21 runs finished; results saved to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
