#  Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exercise:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

#  Initialization
import numpy as np

from scipy.io import loadmat

from displayData import displayData
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

#  =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('ex3data1.mat')  # training data stored in arrays X, y
X = data['X']
y = data['y'].ravel()
m = len(X)

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100]]

displayData(sel)

input('Program paused. Press Enter to continue.\n')

#  ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape(5, 3, order='F') / 10]
y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

np.set_printoptions(precision=6)
print('\nCost: {:f}'.format(J))
print('Expected cost: 2.534819')
print('Gradients:')
print(' {} '.format(grad))
print('Expected gradients:')
print(' [ 0.146561 -0.548558  0.724722  1.398003]')

input('Program paused. Press enter to continue.\n')

#  ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

input('Program paused. Press Enter to continue.\n')

#  ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)

print('\nTraining Set Accuracy: {:.1f}'.format(np.mean(pred == y) * 100))
