import numpy as np

from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lambda_):
    """ computes the cost of using
        theta as the parameter for regularized logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if y.dtype == bool:
        y = y.astype(int)

    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #

    z = X @ theta
    h = sigmoid(z)

    theta_ = np.r_[0, theta[1:]]

    J = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    J += lambda_ * sum(theta_**2) / (2 * m)

    grad = (h - y) @ X / m
    grad += lambda_ * theta_ / m

    #  =============================================================

    return J, grad
