import numpy as np

from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta
    """

    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros(num_iters)

    for iter_ in range(num_iters):

        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        h = X @ theta
        theta -= alpha * (h - y) @ X / m

        # ============================================================

        # Save the cost J in every iteration
        J_history[iter_] = computeCost(X, y, theta)

    return theta, J_history
