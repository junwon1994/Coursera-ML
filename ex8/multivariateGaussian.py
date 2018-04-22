import numpy as np


def multivariateGaussian(X, mu, sigma2):
    """Computes the probability
    density function of the examples X under the multivariate gaussian
    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    as the \sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix)
    """

    k = len(mu)

    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)

    X_ = X - mu
    p = np.exp(-0.5 * np.sum(X_ @ np.linalg.pinv(sigma2) * X_, axis=1)) / ((
        2 * np.pi)**k * np.linalg.det(sigma2))**0.5

    return p
