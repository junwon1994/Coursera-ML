def computeCostMulti(X, y, theta):
    """Compute cost for linear regression with multiple variables
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    h = X.dot(theta)
    J = sum((h - y)**2) / (2 * m)

    # =========================================================================

    return J
