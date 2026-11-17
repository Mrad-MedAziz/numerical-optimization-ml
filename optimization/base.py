import numpy as np # type: ignore
from typing import Callable


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to feature matrix X."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])

def backtracking_line_search(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    grad: np.ndarray,
    alpha: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    min_alpha: float = 1e-10
) -> float:
    """
    Find a step size alpha satisfying the Armijo sufficient decrease condition:
        J(theta - alpha * grad) <= J(theta) - c * alpha * ||grad||^2
    About cost_fn : Callable, it is a function taking theta and returning cost.
    """
    if alpha <= 0 or min_alpha <= 0 or not 0 < c < 1 or not 0 < tau < 1:
        raise ValueError("alpha and min_alpha must be positive; c and tau must lie in (0, 1)")
    cost = cost_fn(theta)
    grad_c =  c * (grad @ grad)
    while alpha >= min_alpha:
        trial_cost = cost_fn(theta - alpha * grad)
        if np.isfinite(trial_cost) and trial_cost <= cost - alpha * grad_c:
            return alpha
        alpha *= tau
    raise RuntimeError("Armijo line search failed to find a finite sufficient-decrease step")
