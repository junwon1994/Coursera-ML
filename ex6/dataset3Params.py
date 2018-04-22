import numpy as np

from sklearn.svm import SVC
from svmPredict import svmPredict


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
    """

    # C, sigma = dataset3Params(X, y, Xval, yval) returns your choice of C and
    # sigma. You should complete this function to return the optimal C and
    # sigma based on a cross-validation set.

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(double(predictions ~= yval))
    #
    values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    pred_error = np.zeros((len(values), len(values)))

    for i, C_ in enumerate(values):
        for j, sigma_ in enumerate(values):
            gamma = 1 / (2 * sigma_**2)
            model = SVC(
                C=C_, kernel='rbf', gamma=gamma, max_iter=200).fit(X, y)
            predictions = svmPredict(model, Xval)
            pred_error[i, j] = np.mean(predictions != yval)

    i, j = np.argwhere(pred_error == np.min(pred_error))[0]

    C = values[i]
    sigma = values[j]
    # =========================================================================

    return C, sigma
