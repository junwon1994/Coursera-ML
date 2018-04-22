import numpy as np

from ex4.computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc


def checkCostFunction(Lambda=0):
    """Creates a collaborative filering problem
    to check your cost function and gradients, it will output the
    analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient
    computations should result in very similar values.
    """

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.ranf(np.shape(Y)) > 0.5] = 0
    R = np.zeros(np.shape(Y))
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.standard_normal(np.shape(X_t))
    Theta = np.random.standard_normal(np.shape(Theta_t))
    n_users = np.size(Y, 1)
    n_movies = np.size(Y, 0)
    n_features = np.size(Theta_t, 1)

    numgrad = computeNumericalGradient(
        lambda t: cofiCostFunc(t, Y, R, n_users, n_movies, n_features, Lambda),
        np.r_[X.ravel(order='F'), Theta.ravel(order='F')])

    cost, grad = cofiCostFunc(
        np.r_[X.ravel(order='F'), Theta.ravel(order='F')],
        Y,
        R,
        n_users,
        n_movies,
        n_features,
        Lambda)

    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          '\nRelative Difference: {:.2g}'.format(diff))
