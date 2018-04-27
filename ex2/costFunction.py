import numpy as np

from sigmoid import sigmoid


def costFunction(theta, X, y):
    """Computes the cost of using theta as the parameter for logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    z = X @ theta
    h = sigmoid(z)

    J = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    grad = (h - y) @ X / m

    # =============================================================

    return J, grad
