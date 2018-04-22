import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_):
    """returns the cost and gradient for the collaborative filtering problem.
    """

    # Unfold the U and W matrices from params
    X = np.reshape(
        params[:num_movies * num_features], (num_movies, num_features),
        order='F')
    Theta = np.reshape(
        params[num_movies * num_features:], (num_users, num_features),
        order='F')

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta
    H = np.dot(X, np.transpose(Theta))
    E = H - Y

    J = np.sum(E**2 * R) / 2
    J += lambda_ * (np.sum(Theta**2) + np.sum(X**2)) / 2

    X_grad = np.dot(E * R, Theta)
    X_grad += lambda_ * X

    Theta_grad = np.dot(np.transpose(E * R), X)
    Theta_grad += lambda_ * Theta
    # =============================================================

    grad = np.r_[X_grad.ravel(order='F'), Theta_grad.ravel(order='F')]

    return J, grad
