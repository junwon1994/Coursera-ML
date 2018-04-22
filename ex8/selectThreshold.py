import numpy as np


def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    for eps in np.linspace(min(pval), max(pval), 1001, dtype='float'):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < eps) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        # If an example x has a low probability p(x) < ε,
        # then it is considered to be an anomaly.
        pred = pval < eps

        # tp is the number of true positives
        # : an anomaly and our algorithm classified it as an anomaly.
        tp = np.sum(np.logical_and(yval == 1, pred == 1))

        # fp is the number of false positives
        # : not an anomaly, but our algorithm classified it as an anomaly.
        fp = np.sum(np.logical_and(yval == 0, pred == 1))

        # fn is the number of false negatives
        # : an anomaly, but our algorithm classified it as not being anomalous.
        fn = np.sum(np.logical_and(yval == 1, pred == 0))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)

        # =============================================================

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = eps

    return bestEpsilon, bestF1
