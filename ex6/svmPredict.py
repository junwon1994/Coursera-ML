def svmPredict(model, X):
    """ returns a vector of predictions using a trained SVM model.

    model is a svm model returned from svmTrain.

    X is a (m x n) matrix where there each example is a row.

    pred is a vector of predictions of {0, 1} values.
    """
    # X: shape (n_samples, n_features)

    # For kernel=”precomputed”,
    # the expected shape of X is [n_samples_test, n_samples_train]

    pred = model.predict(X)

    return pred
