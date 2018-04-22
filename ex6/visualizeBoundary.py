import numpy as np
from matplotlib import pyplot as plt

from plotData import plotData
from svmPredict import svmPredict


def visualizeBoundary(X, y, model):
    """plots a non-linear decision boundary learned by the
    SVM and overlays the data on it"""

    m = np.size(X, 0)

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), m)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), m)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros((m, m))

    for i in range(m):
        this_X = np.c_[X1[:, i], X2[:, i]]  # (863, 2)
        vals[:, i] = svmPredict(model, this_X)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors='blue')
