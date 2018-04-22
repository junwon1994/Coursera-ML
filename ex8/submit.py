import numpy as np
from numpy import sin, cos, reshape

from Submission import Submission
from Submission import sprintf

homework = 'anomaly-detection-and-recommender-systems'

part_names = [
    'Estimate Gaussian Parameters',
    'Select Threshold',
    'Collaborative Filtering Cost',
    'Collaborative Filtering Gradient',
    'Regularized Cost',
    'Regularized Gradient',
]

srcs = [
    'estimateGaussian.py',
    'selectThreshold.py',
    'cofiCostFunc.py',
    'cofiCostFunc.py',
    'cofiCostFunc.py',
    'cofiCostFunc.py',
]


def output(part_id):
    # Random Test Cases
    n_u = 3
    n_m = 4
    n = 5
    X = reshape(sin(range(1, n_m * n + 1)), (n_m, n), order='F')
    Theta = reshape(cos(range(1, n_u * n + 1)), (n_u, n), order='F')
    Y = reshape(sin(range(1, 2 * n_m * n_u + 1, 2)), (n_m, n_u), order='F')
    R = Y > 0.5
    pval = np.r_[abs(Y.ravel(order='F')), 0.001, 1]
    yval = np.r_[R.ravel(order='F'), 1, 0]
    params = np.r_[X.ravel(order='F'), Theta.ravel(order='F')]

    fname = srcs[part_id - 1].rsplit('.', 1)[0]
    mod = __import__(fname, fromlist=[fname], level=0)
    func = getattr(mod, fname)

    if part_id == 1:
        mu, sigma2 = func(X)
        return sprintf('%0.5f ', np.r_[mu, sigma2])
    elif part_id == 2:
        bestEpsilon, bestF1 = func(yval, pval)
        return sprintf('%0.5f ', np.r_[bestEpsilon, bestF1])
    elif part_id == 3:
        J, _ = func(params, Y, R, n_u, n_m, n, 0.0)
        return sprintf('%0.5f ', J)
    elif part_id == 4:
        _, grad = func(params, Y, R, n_u, n_m, n, 0.0)
        return sprintf('%0.5f ', grad)
    elif part_id == 5:
        J, _ = func(params, Y, R, n_u, n_m, n, 1.5)
        return sprintf('%0.5f ', J)
    elif part_id == 6:
        _, grad = func(params, Y, R, n_u, n_m, n, 1.5)
        return sprintf('%0.5f ', grad)


s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = 'An exception of type {0} occured. Messsage:\n{1!r}'
    message = template.format(type(ex).__name__, ex.args)
    print(message)
