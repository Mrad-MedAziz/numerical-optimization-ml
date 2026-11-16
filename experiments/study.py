import time
import numpy as np
from optimization.base import add_bias
from optimization.logistic_model import LogisticRegression


def generate_data(seed: int, n_samples: int = 300):
    """Synthetic overlapping clusters, same generator as benchmark.py."""
    np.random.seed(seed)
    half = n_samples // 2
    X_0 = np.random.randn(half, 2) + np.array([-1.0, -1.0])
    X_1 = np.random.randn(half, 2) + np.array([1.0, 1.0])
    X = np.vstack([X_0, X_1])
    y = np.array([0.0] * half + [1.0] * half)
    return X, y


def reference_ell_star(X, y):
    """Reference optimum from a separate, tighter Newton run."""
    ref = LogisticRegression()
    ref.fit_newton(X, y, max_iter=60, tol=1e-12)
    return ref.log_likelihood_history[-1]


def seeds_comparison():
    """Newton updates to gradient norm < 1e-8; GA iterations to gap < 1e-3 at lr = 0.5."""
    newton_updates, ga_iters = [], []
    for seed in range(20):
        X, y = generate_data(seed)
        ell_star = reference_ell_star(X, y)

        newton = LogisticRegression()
        newton.fit_newton(X, y, max_iter=20, tol=1e-8)
        newton_updates.append(len(newton.log_likelihood_history) - 1)

        ga = LogisticRegression()
        ga.fit_gradient_ascent(X, y, lr=0.5, epochs=2000)
        gap = np.abs(np.array(ga.log_likelihood_history) - ell_star)
        hits = np.where(gap < 1e-3)[0]
        ga_iters.append(int(hits[0]) if hits.size else None)

    reached = [g for g in ga_iters if g is not None]
    print("20 seeds")
    print(f"  Newton updates (gradient norm < 1e-8): {min(newton_updates)} to {max(newton_updates)}")
    print(f"  GA lr = 0.5, iterations to gap < 1e-3: {min(reached)} to {max(reached)}"
          f" (not reached on {ga_iters.count(None)} seeds)")


def learning_rate_sweep():
    """Gradient ascent stops at gradient norm < 1e-5 or the 2000-iteration cap."""
    learning_rates = [0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
    results = {lr: [] for lr in learning_rates}
    for seed in range(20):
        X, y = generate_data(seed)
        for lr in learning_rates:
            model = LogisticRegression()
            model.fit_gradient_ascent(X, y, lr=lr, epochs=2000)
            results[lr].append(len(model.log_likelihood_history) - 1)

    print("\nMedian iterations by learning rate (20 seeds, cap 2000)")
    for lr in learning_rates:
        med = int(np.median(results[lr]))
        note = "  (cap reached: did not converge)" if med >= 1999 else ""
        print(f"  lr = {lr:<4}: {med} iterations{note}")


def timing(repeats: int = 50):
    """Wall-clock time on the benchmark data, medians over repeated runs."""
    X, y = generate_data(42, n_samples=300)
    # Warm-up runs so the first measurement does not include setup costs
    LogisticRegression().fit_newton(X, y, max_iter=20, tol=1e-8)
    LogisticRegression().fit_gradient_ascent(X, y, lr=0.5, epochs=250)

    newton_ms, newton_upd, ga_ms, ga_upd = [], [], [], []
    for _ in range(repeats):
        m = LogisticRegression()
        t0 = time.perf_counter()
        m.fit_newton(X, y, max_iter=20, tol=1e-8)
        newton_ms.append((time.perf_counter() - t0) * 1e3)
        newton_upd.append(len(m.log_likelihood_history) - 1)

        g = LogisticRegression()
        t0 = time.perf_counter()
        g.fit_gradient_ascent(X, y, lr=0.5, epochs=250)
        ga_ms.append((time.perf_counter() - t0) * 1e3)
        ga_upd.append(len(g.log_likelihood_history) - 1)

    def summarize(name, ms, upd):
        q1, med, q3 = np.percentile(ms, [25, 50, 75])
        print(f"  {name}: {upd[0]} updates, median {med:.2f} ms "
              f"(IQR {q1:.2f} to {q3:.2f} ms), {med / upd[0]:.3f} ms per update")

    print(f"\nWall-clock time (benchmark data, {repeats} repeats)")
    summarize("Newton-Raphson", newton_ms, newton_upd)
    summarize("Gradient ascent, lr = 0.5, 250 iterations", ga_ms, ga_upd)


def damping_check(updates: int = 10):
    """Same number of Newton updates for every damping value; gradient stop disabled."""
    np.random.seed(42)
    X_sep = np.vstack([np.random.randn(100, 2) + np.array([-3.0, -3.0]),
                       np.random.randn(100, 2) + np.array([3.0, 3.0])])
    y_sep = np.array([0.0] * 100 + [1.0] * 100)
    X_ov, y_ov = generate_data(42, n_samples=300)

    for label, X, y in [("Overlapping benchmark data", X_ov, y_ov),
                        ("Separable data (centers at +/-3)", X_sep, y_sep)]:
        print(f"\n{label}, {updates} updates")
        for eps in [0.0, 1e-8, 1e-4]:
            m = LogisticRegression()
            m.fit_newton(X, y, max_iter=updates, tol=0.0, eps_reg=eps)
            # Log-likelihood at the final theta (after all updates)
            p = LogisticRegression._sigmoid(add_bias(X) @ m.theta)
            ll = m._cost(y, p)
            print(f"  eps = {eps:<7} norm = {np.linalg.norm(m.theta):.3f}  "
                  f"log-likelihood = {ll:.6f}")


if __name__ == "__main__":
    seeds_comparison()
    learning_rate_sweep()
    timing()
    damping_check()