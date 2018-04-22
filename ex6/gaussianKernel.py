import numpy as np


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """

    # Ensure that x1 and x2 are column vectors
    #     x1 = x1.ravel()
    #     x2 = x2.ravel()
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    gamma = 1 / (2 * sigma**2)
    sim = np.exp(-gamma * (x1 - x2).T @ (x1 - x2))
    # =============================================================

    return sim
