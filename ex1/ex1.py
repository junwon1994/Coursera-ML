#  Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     computeCost.py
#     gradientDescent.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

#  Initialization
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import axes3d

from warmUpExercise import warmUpExercise
from plotData import plotData
from gradientDescent import gradientDescent
from computeCost import computeCost

#  ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
warmUpExercise()

input('Program paused. Press Enter to continue.\n')

#  ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.py
plotData(X, y)

input('Program paused. Press Enter to continue.\n')

#  =================== Part 3: Cost and Gradient descent ===================

X = np.c_[np.ones(m), X]  # Add a column of ones to x
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0  0]\nCost computed = {:.2f}'.format(J))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = computeCost(X, y, np.array([-1, 2]))
print('\nWith theta = [-1  2]\nCost computed = {:.2f}'.format(J))
print('Expected cost value (approx) 54.24')

input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...')
# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print('{}'.format(theta))
print('Expected theta values (approx)')
print('[-3.6303  1.1664]')

# Plot the linear fit
plt.ion()  # keep previous plot visible
plt.plot(X[:, 1], X @ theta, '-')
plt.legend(['Training data', 'Linear regression'], loc='lower right')
plt.ioff()  # dont't overlay any more plots on this figure

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @ theta
print('For population = 35,000, we predict a profit of {:.4f}'.format(
    predict1 * 10000))
predict2 = np.array([1, 7]) @ theta
print('For population = 70,000, we predict a profit of {:.4f}'.format(
    predict2 * 10000))

input('Program paused. Press enter to continue.\n')

#  ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i, theta0_val in enumerate(theta0_vals):
    for j, theta1_val in enumerate(theta1_vals):
        t = np.array([theta0_val, theta1_val])
        J_vals[i, j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# Surface plot
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.jet)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.view_init(30, 50)
ax.axis([10, -10, 4, -1])
ax.xaxis.set_major_locator(LinearLocator(5))

# Contour plot
ax = fig.add_subplot(1, 2, 2)
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.xaxis.set_major_locator(LinearLocator(11))
ax.yaxis.set_major_locator(LinearLocator(11))
ax.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)

plt.tight_layout()
plt.show(block=True)
