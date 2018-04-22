import matplotlib.pyplot as plt


def plotData(X, y):
    """plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    """
    plt.figure()

    # Find Indices of Positive and Negative Examples
    pos = y > 0
    neg = ~pos

    # Plot Examples
    plt.scatter(
        X[pos, 0], X[pos, 1], s=20, c='black', marker='+', edgecolors='black')
    plt.scatter(
        X[neg, 0], X[neg, 1], s=20, c='yellow', marker='o', edgecolors='black')
