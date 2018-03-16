import numpy as np
import sys
sys.path.append('../ex2/')

from Submission import Submission
from Submission import sprintf
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from predict import predict
from ex2.gradientFunctionReg import gradientFunctionReg

homework = 'multi-class-classification-and-neural-networks'

part_names = [
    'Regularized Logistic Regression',
    'One-vs-All Classifier Training',
    'One-vs-All Classifier Prediction',
    'Neural Network Prediction Function',
]

srcs = [
    'lrCostFunction.py',
    'oneVsAll.py',
    'predictOneVsAll.py',
    'predict.py',
]


def output(part_id):
    # Random Test Cases
    X = np.c_[np.ones(20),
              np.exp(1) * np.sin(range(1, 21)),
              np.exp(0.5) * np.cos(range(1, 21))]
    y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(int)

    Xm = np.c_[(-1, -1, -2, -2, 1, 1, 2, 2, -1, -1, -2, -2, 1, 1, -2,
                -2), (-1, -2, -1, -2, 1, 2, 1, 2, 1, 2, 1, 2, -1, -2, -1, -2)]
    ym = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

    theta1 = np.array([0.25, 0.5, -0.5])
    theta2 = np.sin(range(1, 24, 2)).reshape(4, 3, order='F')
    theta3 = np.cos(range(1, 40, 2)).reshape(4, 5, order='F')

    lambda_ = 0.1
    num_labels = 4

    fname = srcs[part_id - 1].rsplit('.', 1)[0]
    mod = __import__(fname, fromlist=[fname], level=0)
    func = getattr(mod, fname)

    if part_id == 1:
        J, grad = func(theta1, X, y, lambda_)
        return sprintf('%0.5f ', np.r_[J, grad])
    elif part_id == 2:
        return sprintf('%0.5f ', func(Xm, ym, num_labels, lambda_))
    elif part_id == 3:
        return sprintf('%0.5f ', func(theta2, Xm))
    elif part_id == 4:
        return sprintf('%0.5f ', func(theta2, theta3, Xm))


s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = 'An exception of type {0} occured. Messsage:\n{1!r}'
    message = template.format(type(ex).__name__, ex.args)
    print(message)
