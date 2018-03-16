import numpy as np
import scipy.io
from scipy.optimize import minimize

from ex3.displayData import displayData
from ex3.predict import predict
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients

#  Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

#  =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex4data1.mat')
X = data['X']
y = data['y'].flatten()
m = np.size(X, 0)

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[:100]

displayData(X[sel])

input('Program paused. Press Enter to continue.')

# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weights = scipy.io.loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# Unroll parameters
nn_params = np.r_[Theta1.flatten(order='F'), Theta2.flatten(order='F')]

#  ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lambda_ = 0.0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %f '
      '\n(this value should be about 0.287629)' % J)

input('\nProgram paused. Press Enter to continue.')

# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.

print('\nChecking Cost Function (with Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lambda_ = 1.0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %f '
      '\n(this value should be about 0.383770)' % J)

input('Program paused. Press Enter to continue.')

#  ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ')
print(g)
print('\n')

input('Program paused. Press Enter to continue.')

#  ================ Part 6: Initializing Parameters ================
#  In this part of the exercise, you will be starting to implement a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.r_[initial_Theta1.flatten(order='F'),
                          initial_Theta2.flatten(order='F')]

#  =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()

input('\nProgram paused. Press Enter to continue.')

#  =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (with Regularization) ... ')

#  Check gradients by running checkNNGradients
lambda_ = 3.0
checkNNGradients(lambda_)

# Also output the costFunction debugging values
debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                            num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (with lambda = %f): %f '
      '\n(for lambda = 3, this value should be about 0.576051)' % (lambda_,
                                                                   debug_J))

input('\nProgram paused. Press Enter to continue.')

#  =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... ')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
options = {'disp': True, 'maxiter': 50}

#  You should also try different values of lambda
lambda_ = 1.0


# Create "short hand" for the cost function to be minimized
def costFunction(p):
    return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels,
                          X, y, lambda_)


# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
result = minimize(
    costFunction, initial_nn_params, method='CG', jac=True, options=options)
nn_params = result.x
cost = result.fun

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
    hidden_layer_size, input_layer_size + 1, order='F')
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
    num_labels, hidden_layer_size + 1, order='F')

input('Program paused. Press Enter to continue.')

#  ================= Part 10: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('\nVisualizing Neural Network... ')

displayData(Theta1[:, 1:])

input('\nProgram paused. Press Enter to continue.')

#  ================= Part 11: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: %f' % (np.mean(pred == y) * 100))
