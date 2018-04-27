import numpy as np
from scipy.optimize import minimize

from lrCostFunction import lrCostFunction


def oneVsAll(X, y, num_labels, lambda_):
    """ trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.c_[np.ones(m), X]

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.

    # Set Initial theta
    initial_theta = np.zeros(n + 1)

    # Set options for minimize
    optimset = {'disp': True, 'maxiter': 500}

    for c in range(num_labels):
        # Run minimize to obtain the optimal theta
        # This function will return theta and the cost
        res = minimize(
            lambda t: lrCostFunction(t, X, y == (c + 1), lambda_),
            initial_theta,
            method='TNC',
            jac=True,
            options=optimset)
        theta = res['x']

        all_theta[c] = theta

    # =========================================================================

    return all_theta
