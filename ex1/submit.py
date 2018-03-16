import numpy as np

from Submission import Submission
from Submission import sprintf

homework = 'linear-regression'

part_names = [
    'Warm up exercise',
    'Computing Cost (for one variable)',
    'Gradient Descent (for one variable)',
    'Feature Normalization',
    'Computing Cost (for multiple variables)',
    'Gradient Descent (for multiple variables)',
    'Normal Equations',
]

srcs = [
    'warmUpExercise.py',
    'computeCost.py',
    'gradientDescent.py',
    'featureNormalize.py',
    'computeCostMulti.py',
    'gradientDescentMulti.py',
    'normalEqn.py',
]


def output(part_id):
    X1 = np.c_[np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)]
    y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])

    X2 = np.c_[X1, X1[:, 1]**0.5, X1[:, 1]**0.25]
    y2 = y1**0.5 + y1

    theta1 = np.array([0.5, -0.5])
    theta2 = np.array([0.1, 0.2, 0.3, 0.4])

    alpha = 0.01
    num_iter = 10

    fname = srcs[part_id - 1].rsplit('.', 1)[0]
    mod = __import__(fname, fromlist=[fname], level=0)
    func = getattr(mod, fname)

    if part_id == 1:
        return sprintf('%0.5f ', func())
    elif part_id == 2:
        return sprintf('%0.5f ', func(X1, y1, theta1))
    elif part_id == 3:
        return sprintf('%0.5f ', func(X1, y1, theta1, alpha, num_iter))
    elif part_id == 4:
        return sprintf('%0.5f ', func(X2[:, 1:4]))
    elif part_id == 5:
        return sprintf('%0.5f ', func(X2, y2, theta2))
    elif part_id == 6:
        return sprintf('%0.5f ', func(X2, y2, -theta2, alpha, num_iter))
    elif part_id == 7:
        return sprintf('%0.5f ', func(X2, y2))


s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = 'An exception of type {0} occured. Messsage:\n{1!r}'
    message = template.format(type(ex).__name__, ex.args)
    print(message)
