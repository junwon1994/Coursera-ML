from numpy import log, dot
from sigmoid import sigmoid
from gradientFunction import gradientFunction


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    h = sigmoid(dot(X, theta))
    J = -(dot(y, log(h)) + dot(1 - y, log(1 - h))) / m
    grad = gradientFunction(theta, X, y)
    # =============================================================

    return J, grad
