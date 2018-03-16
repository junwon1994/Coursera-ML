def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = len(y)
    J = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    h = X @ theta
    error = h - y
    J = error.T @ error / (2 * m)
    # =========================================================================

    return J
