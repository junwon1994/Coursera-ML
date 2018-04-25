import matplotlib.pyplot as plt


def plotData(x, y):
    """plots the data points and gives the figure axes labels of
    population and profit.
    """

    plt.figure()  # open a new figure window

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.

    plt.plot(x, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show(block=False)

    # ============================================================
