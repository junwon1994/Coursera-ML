from numpy import dot
from sigmoid import sigmoid


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter
    for regularized logistic regression and the gradient
    of the cost w.r.t. to the parameters.
    """

    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    h = sigmoid(dot(X, theta))
    grad = dot(X.T, h - y) / m
    # =============================================================

    return grad
