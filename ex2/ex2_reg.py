# Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import use
from scipy.optimize import minimize

from ml import mapFeature, plotData, plotDecisionBoundary

from costFunctionReg import costFunctionReg
from predict import predict

use('TkAgg')


def optimize(lambda_):
    args = (X, y, lambda_)
    options = {'gtol': 1e-4, 'disp': True}

    result = minimize(
        costFunctionReg,
        initial_theta,
        method=None,
        jac=True,
        args=args,
        options=options)

    return result


# Plot Boundary
def plotBoundary(theta, X, y):
    plotDecisionBoundary(theta, X, y)
    plt.title(r'$\lambda$ = ' + str(lambda_))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc='upper right', shadow=True, numpoints=1)
    plt.xlim(-1, 1.5)
    plt.ylim(-0.8, 1.2)
    plt.show(block=False)


# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

# data = pd.read_csv('ex2data2.txt', header=None, names=[1, 2, 3])
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(loc='upper right', shadow=True)
plt.xlim(-1, 1.5)
plt.ylim(-0.8, 1.2)
plt.show(block=False)

input('Program paused. Press Enter to continue...')

# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = pd.DataFrame(X)
X = X.apply(mapFeature, axis=1)
# convert back to numpy ndarray
X = X.values

# Initialize fitting parameters
m, n = X.shape
initial_theta = np.zeros(n)

# Set regularization parameter lambda to 1
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, _ = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): %f' % cost)

# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary
result = optimize(lambda_)
theta = result.x
cost = result.fun

# Print to screen
print('lambda = ' + str(lambda_))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])

input('Program paused. Press Enter to continue...')

plotBoundary(theta, X, y)

# Compute accuracy on our training set
p = predict(theta, X)
acc = np.mean(np.where(p == y, 1, 0)) * 100
print('Train Accuracy: %f' % acc)

input('Program paused. Press Enter to continue...')

# ============= Part 3: Optional Exercises =============

for lambda_ in (1, 0, 100):
    result = optimize(lambda_)
    theta = result.x
    p = predict(theta, X)
    acc = np.mean(p == y) * 100
    print('lambda = ' + str(lambda_))
    print('Train Accuracy: %f' % acc)
    plotBoundary(theta, X, y)
input('Program paused. Press Enter to continue...')
