#  Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

#  Initialization
import numpy as np

from scipy.io import loadmat

from displayData import displayData
from predict import predict

#  Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

#  =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y'].ravel()
m = len(X)

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[:100]

displayData(X[sel])

input('Program paused. Press Enter to continue.\n')

#  ================ Part 2: Loading Pameters ================
#  In this part of the exercise, we load some pre-initialized
#  neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weights = loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

#  ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: {:.1f}'.format(np.mean(pred == y) * 100))

input('Program paused. Press Enter to continue...')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in range(m):
    # Display
    print('\nDisplaying Example Image')
    displayData(X[rp[i]])

    pred = predict(Theta1, Theta2, X[rp[i]])
    print('\nNeural Network Prediction: {:d} (digit {:d})'.format(
        pred, pred % num_labels))

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
