from costFunction import costFunction
from numpy import dot


def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter
    for regularized logistic regression and the gradient
    of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    J, grad = costFunction(theta, X, y)
    theta[0] = 0  # We should not regularize first theta.
    J += lambda_ * dot(theta.T, theta) / (2 * m)
    grad += lambda_ * theta / m
    # =============================================================

    return J, grad
