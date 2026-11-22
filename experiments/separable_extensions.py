from pathlib import Path
import csv
import json
import platform
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.separable import datasets, reference_solution, run_one, measurements
from optimization.separable_logistic import (logistic_gradient, sigmoid,
                                             regularized_loss, regularized_gradient)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "separable"
FIGURES = ROOT / "figures" / "separable"
STEPS = 100000
PENALTIES = [0.0, 1e-2, 1e-3, 1e-4]
TOLERANCE = 1e-10


def save_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def regularized_reference(X, y, penalty, tolerance=1e-12):
    if penalty <= 0:
        raise ValueError("A positive penalty is required for the finite reference")
    w = np.zeros(X.shape[1])
    for step in range(100):
        gradient = regularized_gradient(X, y, w, penalty)
        if np.linalg.norm(gradient) < tolerance:
            return w
        q = sigmoid(-y * (X @ w))
        hessian = X.T @ ((q * (1 - q))[:, None] * X) / len(y)
        hessian += penalty * np.eye(X.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        current = regularized_loss(X, y, w, penalty)
        step_size = 1.0
        while step_size >= 1e-12:
            trial = w - step_size * direction
            if regularized_loss(X, y, trial, penalty) <= current - 1e-4 * step_size * (gradient @ direction):
                w = trial
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Regularized Newton line search failed")
    raise RuntimeError("Regularized reference did not meet its gradient tolerance")


def run_regularized(X, y, w_hat, penalty, n_steps=STEPS, tolerance=TOLERANCE):
    B = np.linalg.norm(X, ord=2)**2 / (4 * len(y))
    lr = 0.5 / (B + penalty)
    w = np.zeros(X.shape[1])
    target = w_hat / np.linalg.norm(w_hat)
    maximum_margin = 1 / np.linalg.norm(w_hat)
    checkpoints = {0, n_steps}
    if n_steps > 0:
        checkpoints |= set(np.geomspace(1, n_steps, 150).astype(int))
    history = []
    previous = regularized_loss(X, y, w, penalty)
    reason = "iteration_budget"
    for step in range(n_steps + 1):
        gradient = regularized_gradient(X, y, w, penalty)
        objective = regularized_loss(X, y, w, penalty)
        if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("Nonfinite regularized iterate")
        if objective > previous + 1e-12:
            raise RuntimeError("Regularized objective increased")
        if penalty > 0 and 0.5 * penalty * (w @ w) > np.log(2) + 1e-12:
            raise RuntimeError("Iterate escaped its coercive initial sublevel set")
        previous = objective
        if np.linalg.norm(gradient) < tolerance:
            reason = "gradient_tolerance"
        if step in checkpoints or reason == "gradient_tolerance":
            row = dict(penalty=penalty, lr=lr, update=step,
                       **measurements(X, y, w, gradient, target, maximum_margin),
                       penalized_objective=float(objective),
                       penalty_term=float(0.5 * penalty * (w @ w)),
                       logistic_gradient_norm=float(np.linalg.norm(logistic_gradient(X, y, w))),
                       w1=float(w[0]), w2=float(w[1]))
            history.append(row)
        if reason == "gradient_tolerance" or step == n_steps:
            break
        w = w - lr * gradient
    final = dict(row, stop_reason=reason, tolerance=tolerance)
    if penalty > 0:
        reference = regularized_reference(X, y, penalty)
        reference_gradient = np.linalg.norm(regularized_gradient(X, y, reference, penalty))
        bound = np.linalg.norm(gradient)**2 / (2 * penalty)
        gap = objective - regularized_loss(X, y, reference, penalty)
        if gap < -1e-12 or gap > bound + 1e-12:
            raise RuntimeError("Objective discrepancy exceeds the strong-convexity bound")
        # Each distance is bounded by its gradient norm divided by the penalty.
        distance = np.linalg.norm(w - reference)
        if distance > (np.linalg.norm(gradient) + reference_gradient) / penalty + 1e-10:
            raise RuntimeError("GD and Newton disagree beyond their gradient certificates")
        final.update(reference_w1=float(reference[0]), reference_w2=float(reference[1]),
                     reference_gradient_norm=float(reference_gradient),
                     distance_to_reference=float(distance), objective_gap=float(gap),
                     objective_gap_bound=float(bound),
                     sublevel_norm_bound=float(np.sqrt(2 * np.log(2) / penalty)))
    else:
        final.update(reference_w1="", reference_w2="", reference_gradient_norm="",
                     distance_to_reference="", objective_gap="", objective_gap_bound="",
                     sublevel_norm_bound="")
    return history, final


def fixed_step_clearance():
    samples = {(family, seed): (X, y) for family, seed, X, y in datasets()}
    history, final = [], []
    for seed in [42, 0, 1]:
        families = ["gaussian_sign", "gaussian_gap"]
        bounds = {name: np.linalg.norm(samples[name, seed][0], ord=2)**2 / (4 * len(samples[name, seed][1]))
                  for name in families}
        lr = 0.5 / max(bounds.values())
        for name in families:
            X, y = samples[name, seed]
            w_hat, checks = reference_solution(X, y)
            assert 0 < lr < 2 / bounds[name]
            metadata = dict(dataset=name, seed=seed, lr=lr, B=bounds[name])
            rows, _, endpoint = run_one(X, y, w_hat, lr, metadata,
                                        n_steps=STEPS, tolerances=[])
            history.extend(rows)
            final.append(endpoint)
        print(f"Finished fixed-step clearance pair, seed={seed}", flush=True)
    save_csv("clearance_fixed_step_results.csv", final)
    save_csv("clearance_fixed_step_trajectories.csv", history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, seed in zip(axes, [42, 0, 1]):
        for family, label in [("gaussian_sign", "Original"), ("gaussian_gap", "Added clearance")]:
            rows = [r for r in history if r["seed"] == seed and r["dataset"] == family and r["update"] > 0]
            ax.semilogx([r["update"] for r in rows], [r["direction_error"] for r in rows], label=label)
        ax.set_title(f"Seed {seed}; shared step {next(r['lr'] for r in final if r['seed'] == seed):.4g}")
        ax.set_xlabel("Updates (log scale)")
        ax.set_ylabel("Distance to SVM direction")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "clearance_fixed_step.png", dpi=160)
    plt.close(fig)


def regularization_comparison():
    _, _, X, y = next(row for row in datasets() if row[0] == "gaussian_sign" and row[1] == 42)
    w_hat, _ = reference_solution(X, y)
    history, final = [], []
    for penalty in PENALTIES:
        rows, endpoint = run_regularized(X, y, w_hat, penalty)
        history.extend(rows)
        final.append(dict(dataset="gaussian_sign", seed=42, **endpoint))
        print(f"Finished penalty={penalty:g}: {endpoint['update']} updates; {endpoint['stop_reason']}", flush=True)
    save_csv("regularization_results.csv", final)
    save_csv("regularization_trajectories.csv", history)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("loss", "Unpenalized logistic loss"), ("weight_norm", "Weight norm"),
               ("gradient_norm", "Gradient norm of optimized objective"),
               ("direction_error", "Distance to unregularized SVM direction")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for penalty in PENALTIES:
            rows = [r for r in history if r["penalty"] == penalty and r["update"] > 0]
            ax.semilogx([r["update"] for r in rows], [r[metric] for r in rows], label=f"penalty={penalty:g}")
        if metric in ["loss", "gradient_norm"]:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Updates (log scale)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Seed 42; step = 0.5/(B + penalty); positive penalties stop at gradient norm < 1e-10")
    fig.tight_layout()
    fig.savefig(FIGURES / "regularization_comparison.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fixed_step_clearance()
    regularization_comparison()
    settings = dict(steps=STEPS, seeds=[42, 0, 1], regularization_seed=42,
                    penalties=PENALTIES, regularized_gradient_tolerance=TOLERANCE,
                    clearance_step_rule="0.5 / max(B_original, B_clearance), shared within each pair",
                    regularized_step_rule="0.5 / (B + penalty)",
                    python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, elapsed_seconds=time.monotonic() - start)
    (OUTPUT / "extensions_configuration.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
