import matplotlib.pyplot as plt


def plotData(X, y):
    """Plots the data points X and y into a new figure
    """

    # Create New Figure
    plt.figure(figsize=(8, 5))
    plt.ion()

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #

    pos = X[y == 1]
    neg = X[y == 0]

    plt.plot(pos[:, 0], pos[:, 1], 'k+', markerfacecolor='black')
    plt.plot(neg[:, 0], neg[:, 1], 'ko', markerfacecolor='yellow')
    plt.show(block=False)

    # =========================================================================

    plt.ioff()
