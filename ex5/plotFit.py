import matplotlib.pyplot as plt
import numpy as np

from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """plots the learned polynomial fit with power p
    and feature normalization (mu, sigma).
    """

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.c_[np.ones(np.size(X_poly, 0)), X_poly]

    # Plot
    plt.plot(x, X_poly @ theta.T, '--', lw=2)
