import numpy as np


def linearRegCostFunction(X, y, theta, lambda_):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ===================================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
    h = X @ theta.T
    delta = h - y

    J = sum(delta**2) / (2 * m)
    J += lambda_ * sum(theta[1:]**2) / (2 * m)

    grad = X.T @ delta / m
    grad += lambda_ * np.r_[0, theta[1:]] / m
    # =========================================================================

    return J, grad
