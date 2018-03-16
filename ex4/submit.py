import numpy as np

from Submission import Submission
from Submission import sprintf

homework = 'neural-network-learning'

part_names = [
    'Feedforward and Cost Function',
    'Regularized Cost Function',
    'Sigmoid Gradient',
    'Neural Network Gradient (Backpropagation)',
    'Regularized Gradient',
]

srcs = [
    'nnCostFunction.py',
    'nnCostFunction.py',
    'sigmoidGradient.py',
    'nnCostFunction.py',
    'nnCostFunction.py',
]


def output(part_id):
    # Random Test Cases
    X = 3 * np.sin(range(1, 31)).reshape(3, 10, order='F')

    Xm = np.sin(range(1, 33)).reshape(16, 2, order='F') / 5
    ym = np.arange(1, 17) % 4 + 1

    t1 = np.sin(range(1, 24, 2)).reshape(4, 3, order='F')
    t2 = np.cos(range(1, 40, 2)).reshape(4, 5, order='F')
    theta = np.r_[t1.flatten(order='F'), t2.flatten(order='F')]

    lambda_1 = 0.0
    lambda_2 = 1.5

    input_ = 2
    hidden = 4
    num_labels = 4

    fname = srcs[part_id - 1].rsplit('.', 1)[0]
    mod = __import__(fname, fromlist=[fname], level=0)
    func = getattr(mod, fname)

    if part_id == 1:
        J, grad = func(theta, input_, hidden, num_labels, Xm, ym, lambda_1)
        return sprintf('%0.5f ', J)
    elif part_id == 2:
        J, grad = func(theta, input_, hidden, num_labels, Xm, ym, lambda_2)
        return sprintf('%0.5f ', J)
    elif part_id == 3:
        g = func(X)
        return sprintf('%0.5f ', g)
    elif part_id == 4:
        J, grad = func(theta, input_, hidden, num_labels, Xm, ym, lambda_1)
        return sprintf('%0.5f ', np.r_[J, grad])
    elif part_id == 5:
        J, grad = func(theta, input_, hidden, num_labels, Xm, ym, lambda_2)
        return sprintf('%0.5f ', np.r_[J, grad])


s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = 'An exception of type {0} occured. Messsage:\n{1!r}'
    message = template.format(type(ex).__name__, ex.args)
    print(message)
