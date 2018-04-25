import numpy as np


def mapFeature(X1, X2):
    """Maps the two input features quadratic features
    used in the regularization exercise.
    """

    degree = 6
    out = np.ones((X1.size, sum(range(degree + 2))))

    end = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, end] = X1**(i - j) * X2**j
            end += 1

    return out
