import numpy as np

from linearKernel import linearKernel
from sklearn import svm


def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=5):
    """ trains an SVM classifier and returns trained model.

    X is the matrix of training examples.
    Each row is a training example, and the jth column hold the jth feature.

    y is a vector.
    Each element is 1 for positive examples or 0 for negative examples.

    C is the standard SVM regularization parameter.

    tol is a tolerance value.
    It is used for determining equality of floating point numbers.

    max_passes is the number of iterations.
    It controls over the dataset (without changes to alpha).
    """

    if kernelFunction.__name__ == 'linearKernel':
        clf = svm.SVC(C=C, kernel='linear', tol=tol, max_iter=max_passes)
        model = clf.fit(X, y)
    else:
        clf = svm.SVC(C=C, kernel='precomputed', tol=tol, max_iter=max_passes)
        model = clf.fit(gaussianKernelGramianMatrix(X, X, kernelFunction), y)

    return model


def gaussianKernelGramianMatrix(X1, X2, kernelFunction):
    Gramian = np.zeros((np.size(X1, 0), np.size(X2, 0)))

    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            Gramian[i, j] = np.asscalar(kernelFunction(x1, x2))

    return Gramian
