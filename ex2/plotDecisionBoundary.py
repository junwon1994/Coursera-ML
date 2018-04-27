import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    """Plots the data points with + for the positive examples and o for the
    negaive examples. X is assumed to be a either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """
    # Plot Data
    plotData(X[:, 1:], y)
    plt.ion()

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[2] * plot_x + theta[0])

        # Plot
        plt.plot(plot_x, plot_y)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the gird
        for i, u_ in enumerate(u):
            for j, v_ in enumerate(v):
                z[i, j] = mapFeature(u_, v_) @ theta

        z = z.T  # important to transpose z before calling contour
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, levels=[0], linewidths=2)

    plt.ioff()
