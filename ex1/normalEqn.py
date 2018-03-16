import numpy as np


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    # ---------------------- Sample Solution ----------------------
    Gramian = X.T @ X
    moment = X.T @ y
    theta = np.linalg.pinv(Gramian) @ moment
    # -------------------------------------------------------------

    return theta


# ============================================================
