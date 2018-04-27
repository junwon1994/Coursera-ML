#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exercise:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

#  Initialization
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from plotData import plotData
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

#  Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

#  ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) and o '
      'indicating (y = 0) examples.')

plotData(X, y)

# Put some labels
plt.ion()
# Labels and Legned
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(['Admitted', 'Not admitted'], loc='upper right')
plt.ioff()

input('\nProgram paused. Press Enter to continue.\n')

#  ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.py

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

np.set_printoptions(precision=4)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ')
print(' {} '.format(grad))
print('Expected gradients (approx):\n [ -0.1000 -12.0092 -11.2628]')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

np.set_printoptions(precision=3)
print('\nCost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
print(' {} '.format(grad))
print('Expected gradients (approx):\n [0.043 2.566 2.647]')

input('\nProgram paused. Press Enter to continue.\n')

#  ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for minimize
optimset = {'disp': True, 'maxiter': 400}

#  Run minimize to obtain the optimal theta
#  This function will return theta and the cost
res = minimize(
    lambda t: costFunction(t, X, y),
    initial_theta,
    method='TNC',
    jac=True,
    options=optimset)
theta = res['x']
cost = res['fun']

# Print theta to screen
np.set_printoptions(precision=3)
print('Cost at theta found by minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: ')
print(' {} '.format(theta))
print('Expected theta (approx):')
print(' [-25.161   0.206   0.201]')

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels
plt.ion()
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(
    ['Admitted', 'Not admitted', 'Decision Boundary'], loc='upper right')
plt.ioff()

input('\nProgram paused. Press Enter to continue.\n')

#  ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.py

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with scores 45 and 85, we predict an admission '
      'probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: {:.1f}'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0')
