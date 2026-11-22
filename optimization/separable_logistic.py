import numpy as np


def logistic_loss (X,y,w):
    return np.mean(np.logaddexp(0, - y * (X @ w)))


def logistic_gradient (X,y,w):
    s = y * ( X @ w)
    return -(X.T @ (y * sigmoid(-s))) / X.shape[0]


def sigmoid(z):
    exp_negative_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1 + exp_negative_abs) , exp_negative_abs / (1.0 + exp_negative_abs) )


def gradient_descent(X, y, w_hat, lr=0.1, n_steps=100000, print_every=0):
    w = np.zeros(X.shape[1])

    losses = []
    gradient_norms = []
    weight_norms = []
    directions = []
    direction_errors = []
    margins = []

    svm_direction = w_hat / np.linalg.norm(w_hat)
    # Record the starting point, before any updates.
    losses.append(logistic_loss(X, y, w))
    gradient_norms.append(np.linalg.norm(logistic_gradient(X, y, w)))
    weight_norms.append(np.linalg.norm(w))
    if print_every:
        print("Starting loss:", losses[-1])

    for step in range(n_steps):
        gradient = logistic_gradient(X, y, w)
        w = w - lr * gradient

        loss = logistic_loss(X, y, w)
        gradient_norm = np.linalg.norm(logistic_gradient(X, y, w))
        weight_norm = np.linalg.norm(w)
        direction = w / weight_norm
        margin = np.min(y * (X @ w)) / weight_norm
        direction_error = np.linalg.norm(direction - svm_direction)

        losses.append(loss)
        gradient_norms.append(gradient_norm)
        weight_norms.append(weight_norm)
        directions.append(direction)
        margins.append(margin)
        direction_errors.append(direction_error)

        if print_every and (step + 1) % print_every == 0:
            print("Updates:", step + 1)
            print("Loss:", loss)
            print("Gradient norm:", gradient_norm)
            print("Weight norm:", weight_norm)
            print("Direction", direction)
            print("Distance to SVM direction:", direction_error)
            print("Geometric margin:", margin)
            print()

    if print_every:
        print("Final weights:", w)
    return w, losses, gradient_norms, weight_norms, directions, direction_errors, margins


def max_margin_solution(X, y):
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("hard_margin_svm")
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    model.Params.BarConvTol = 1e-10

    # The weights can be negative. No intercept is added.
    w = model.addMVar(X.shape[1], lb=-GRB.INFINITY)
    model.setObjective(0.5 * (w @ w), GRB.MINIMIZE)

    constraints = []
    for i in range(len(y)):
        constraint = model.addConstr(y[i] * (X[i] @ w) >= 1)
        constraints.append(constraint)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimum. Status: {model.Status}")

    w_hat = w.X
    minimum_margin = np.min(y * (X @ w_hat))
    if minimum_margin < 1 - 1e-6:
        raise RuntimeError("SVM margin constraints are not satisfied")
    multipliers = np.array([constraint.Pi for constraint in constraints])
    return w_hat, multipliers
