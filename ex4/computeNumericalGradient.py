import numpy as np


def computeNumericalGradient(J, theta):
    """
        Computes the numerical gradient of the function J around theta.
        Calling y = J(theta) should return the function value at theta.
    """
    # Notes: The following code implements numerical gradient checking, and
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical
    #        approximation of) the partial derivative of J with respect to the
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    #        be the (approximately) the partial derivative of J with respect
    #        to theta(i).)
    m = len(theta)
    numgrad = np.zeros(m)
    perturb = np.zeros(m)
    e = 1e-4

    for p in range(theta.size):

        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad
