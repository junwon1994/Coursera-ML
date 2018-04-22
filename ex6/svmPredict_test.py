import numpy as np


def svmPredict_test(model, X):
    """ returns a vector of predictions using a trained SVM model.

    model is a svm model returned from svmTrain.

    X is a (m x n) matrix where there each example is a row.

    pred is a vector of predictions of {0, 1} values.
    """

    # Dataset
    m = np.size(X, 0)
    p = np.zeros(m)
    pred = np.zeros(m)

    if model.kernelFunction.__name__ == 'linearKernel':
        p = model.b + X @ model.w
    else:
        X1 = np.sum(X**2, axis=1).reshape(-1, 1)
        X2 = np.sum(model.X**2, axis=1).T
        K = X1 + X2 - 2 * X @ model.X.T
        K = model.kernelFunction(1, 0)**K
        K = model.y * K
        K = model.alphas * K
        p = np.sum(K, axis=1)

    pred[p >= 0] = 1
    pred[p < 0] = 0

    return pred
