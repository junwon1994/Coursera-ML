import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

#  Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

#  =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y'].flatten()
Xval = data['Xval']
yval = data['yval'].flatten()
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

# m = Number of examples
m = np.size(X, 0)

# Plot training data
plt.figure()
plt.scatter(X, y, marker='x', s=60, edgecolor='r', color='r', lw=1.5)
plt.xlabel('Change in water level (x)')  # Set the x-axis label
plt.ylabel('Water flowing out of the dam (y)')  # Set the y-axis label
plt.xlim(-50, 40)
plt.ylim(-5, 40)
plt.show(block=False)

input('Program paused. Press Enter to continue...')

#  =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.

theta = np.array([1, 1])
J, _ = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n'
      % J)

input('Program paused. Press Enter to continue...')

#  =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.

theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [%f %f] \n'
      '(this value should be about [-15.303016 598.250744])\n' % (grad[0],
                                                                  grad[1]))

input('Program paused. Press Enter to continue...')

#  =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#

#  Train linear regression with lambda = 0
lambda_ = 0
theta = trainLinearReg(np.c_[np.ones(m), X], y, lambda_)

#  Plot fit over the data
plt.figure()
plt.scatter(X, y, marker='x', s=60, edgecolor='r', color='r', lw=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlim(-50, 40)
plt.ylim(-5, 40)

plt.plot(X, np.c_[np.ones(m), X] @ theta.T, '--', lw=2.0)
plt.show(block=False)

input('Program paused. Press Enter to continue...')

#  =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
#

lambda_ = 0
error_train, error_val = learningCurve(np.c_[np.ones(m), X], y,
                                       np.c_[np.ones(np.size(Xval, 0)), Xval],
                                       yval, lambda_)

plt.figure()
plt.plot(range(m), error_train, color='b', lw=0.5)
plt.plot(range(m), error_val, color='g', lw=0.5)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'], loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.show(block=False)

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d  \t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

input('Program paused. Press Enter to continue...')

#  =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.c_[np.ones(m), X_poly]  # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.c_[np.ones(np.size(X_poly_test, 0)), X_poly_test]  # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.c_[np.ones(np.size(X_poly_val, 0)), X_poly_val]  # Add Ones

print('Normalized Training Example 1:')
print(X_poly[0])

input('\nProgram paused. Press Enter to continue.')

#  =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.

lambda_ = 0
theta = trainLinearReg(X_poly, y, lambda_)

# Plot training data and fit
plt.figure()
plt.scatter(X, y, marker='x', s=60, edgecolor='r', color='r', lw=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
plt.show(block=False)

plt.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.plot(range(m), error_train, color='b', lw=0.5)
plt.plot(range(m), error_val, color='g', lw=0.5)
plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend(['Train', 'Cross Validation'], loc='upper right')
plt.show(block=False)

print('Polynomial Regression (lambda = %f)\n' % lambda_)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d  \t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

input('Program paused. Press Enter to continue...')

#  =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val,
                                                     yval)

plt.figure()
plt.plot(lambda_vec, error_train, color='b', lw=0.5)
plt.plot(lambda_vec, error_val, color='g', lw=0.5)
plt.legend(['Train', 'Cross Validation'], loc='upper right')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show(block=False)

print('lambda\t\tTrain Error\tValidation Error')
for i, lambda_ in enumerate(lambda_vec):
    print('%f\t%f\t%f' % (lambda_, error_train[i], error_val[i]))

input('Program paused. Press Enter to continue...')
