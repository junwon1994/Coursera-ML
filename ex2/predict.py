import numpy as np

from sigmoid import sigmoid


def predict(theta, X):
    """Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

    m = len(X)  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's
    #

    z = X @ theta
    h = sigmoid(z)

    p[h >= 0.5] = 1

    # =========================================================================

    return p
