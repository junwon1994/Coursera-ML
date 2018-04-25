import numpy as np


def sigmoid(z):
    """Computes the sigmoid of z."""

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).

    g = 1 / (1 + np.exp(-z))

    # =============================================================

    return g
