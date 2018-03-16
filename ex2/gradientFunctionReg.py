from gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter
    for regularized logistic regression
    and the gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    grad = gradientFunction(theta, X, y)
    theta[0] = 0  # We should not regularize first theta.
    grad += lambda_ * theta / m
    # =============================================================

    return grad
