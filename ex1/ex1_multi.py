#  Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     featureNormalize.py
#     computeCostMulti.py
#     gradientDescentMulti.py
#     normalEqn.py
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

#  Initialization
import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

#  ================ Part 1: Feature Normalization ================

print('Loading data ...')

#  Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = len(y)

# Print out some data points
np.set_printoptions(precision=0)
print('First 10 examples from the dataset: ')
print('{}'.format(data[:10]))

input('Program paused. Press Enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.c_[np.ones(m), X]

# ================ Part 2: Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show(block=False)

# Display gradient descent's result
np.set_printoptions(precision=6)
print('Theta computed from gradient descent: ')
print(' {} '.format(theta))
print()

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = np.array([1, 1650, 3]) @ theta

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house '
      '(using gradient descent):\n ${:f}'.format(price))

input('Program paused. Press Enter to continue.\n')

# ================ Part 3: Normal Equations ================

print('Solving with normal equations...')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.py
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#
#

#  Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = len(y)

# Add intercept term to X
X = np.c_[np.ones(m), X]

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: ')
print(' {} '.format(theta))
print()

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

price = np.array([1, 1650, 3]) @ theta

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house '
      '(using normal equation):\n ${:f}'.format(price))
