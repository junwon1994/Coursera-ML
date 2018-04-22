import numpy as np


def normalizeRatings(Y, R):
    """normalized Y so that each movie has a rating of 0 on average,
    and returns the mean rating in Ymean.
    """

    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = R[i] == 1
        y = Y[i, idx]

        Ymean[i] = np.mean(y)
        Ynorm[i, idx] = y - Ymean[i]

    return Ynorm, Ymean
