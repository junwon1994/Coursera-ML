import numpy as np


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
    """
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    # ---------------------- Sample Solution ----------------------
    Gramian = X.T.dot(X)
    moment = X.T.dot(y)
    theta = np.linalg.pinv(Gramian).dot(moment)
    # -------------------------------------------------------------

    # ============================================================

    return theta
