#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

#  Initialization
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

#  Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

plotData(X, y)

# Put some labels
plt.ion()

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'], loc='upper right')
plt.ioff()

#  =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

np.set_printoptions(precision=4, suppress=True)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(' {} '.format(grad[:5]))
print('Expected gradients (approx) - first five values only:')
print(' [0.0085 0.0188 0.0001 0.0503 0.0115]')

input('\nProgram paused. Press enter to continue.\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:')
print(' {} '.format(grad[:5]))
print('Expected gradients (approx) - first five values only:')
print(' [0.3460 0.1614 0.1948 0.2269 0.0922]')

input('\nProgram paused. Press enter to continue.\n')

#  ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# Set Options
optimset = {'disp': True, 'maxiter': 400}

# Optimize
res = minimize(
    lambda t: costFunctionReg(t, X, y, lambda_),
    initial_theta,
    method='TNC',
    jac=True,
    options=optimset)
theta = res['x']
J = res['fun']

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.ion()
plt.title('lambda = {:g}'.format(lambda_))

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
plt.ioff()

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: {:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

input('\nProgram paused. Press enter to continue.\n')
