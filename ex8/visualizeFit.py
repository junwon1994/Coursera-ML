import matplotlib.pyplot as plt
import numpy as np

from math import isinf

from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    """
    This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """

    n = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(n, n)
    Z = multivariateGaussian(
        np.c_[X1.ravel(order='F'), X2.ravel(order='F')], mu, sigma2)
    Z = np.reshape(Z, X1.shape, order='F')

    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bx', markersize=3, markeredgewidth=0.5)
    # Do not plot if there are infinities
    if not isinf(np.sum(Z)):
        plt.contour(X1, X2, Z, 10**np.arange(-20, 0, 3, dtype='float'))
