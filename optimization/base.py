import numpy as np # type: ignore
from typing import Callable, Tuple


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend an intercept column of ones to feature matrix X."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X]) #Merges the first parameter (first matrix) with the second one to obtain a matrix of size n * 2

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
    cost = cost_fn (theta)
    grad_c =  c * (grad @ grad)
    while ((cost_fn(theta - alpha * grad ) > cost - alpha * grad_c ) and (alpha > min_alpha)) :
        alpha *= tau
    return alpha
