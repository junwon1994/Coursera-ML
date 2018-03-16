import numpy as np

from scipy.optimize import minimize

from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lambda_):
    """trains linear regression using the dataset (X, y) and
    regularization parameter lambda. Returns the trained parameters theta.
    """

    # Initialize Theta
    initial_theta = np.zeros(np.size(X, 1))

    # Create "short hand" for the cost function to be minimized
    def costFunction(t):
        return linearRegCostFunction(X, y, t, lambda_)

    # Now, costFUnction is a function that takes in only one argument
    options = {'maxiter': 200, 'disp': True}

    # Minimize using minimize(method='CG')
    theta = minimize(
        costFunction, initial_theta, method='CG', jac=True, options=options).x

    return theta
