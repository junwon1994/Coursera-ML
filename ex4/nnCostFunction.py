import numpy as np
import pandas as pd

from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                   X, y, lambda_):
    """ computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
    """

    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        hidden_layer_size, input_layer_size + 1, order='F')  # (25, 401)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        num_labels, hidden_layer_size + 1, order='F')  # (10, 26)

    # Setup some useful variables
    m = len(X)
    y = pd.get_dummies(y).as_matrix()

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial
    #         derivatives of the cost function with respect to Theta1 and
    #         Theta2 in Theta1_grad and Theta2_grad, respectively.
    #         After implementing Part 2, you can check that
    #         your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector
    #               into a binary vector of 1's and 0's to be used with
    #               the neural network cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it
    #               for the first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for backpropagation.
    #               That is, you can compute the gradients
    #               for the regularization separately and then add them
    #               to Theta1_grad and Theta2_grad from Part 2.
    #

    # Feedforward the neural network...
    a1 = np.c_[np.ones(m), X]  # (5000, 401)

    z2 = a1 @ Theta1.T  # (5000, 401) @ (401, 25) = (5000, 25)
    a2 = np.c_[np.ones(len(z2)), sigmoid(z2)]  # (5000, 26)

    z3 = a2 @ Theta2.T  # (5000, 26) @ (26, 10) = (5000, 10)
    a3 = sigmoid(z3)  # (5000, 10)

    # Computing cost...
    J = -np.mean(np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3), axis=1))

    # Computing regularized cost...
    J += lambda_ * (sum(np.sum(np.square(Theta1[:, 1:]), axis=1)) +
                    sum(np.sum(np.square(Theta2[:, 1:]), axis=1))) / (2 * m)

    # Computing δ(del) and ∆(delta)...
    del3 = a3 - y  # (5000, 10)
    delta2 = del3.T @ a2  # (10, 26)

    del2 = del3 @ Theta2 * sigmoidGradient(np.c_[np.ones(len(z2)), z2])
    delta1 = del2[:, 1:].T @ a1  # (25, 401)

    # Computing gradient...
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m

    # Computing regularized gradient...
    Theta1_grad += lambda_ * np.c_[np.zeros(len(Theta1)), Theta1[:, 1:]] / m
    Theta2_grad += lambda_ * np.c_[np.zeros(len(Theta2)), Theta2[:, 1:]] / m
    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradient
    grad = np.r_[Theta1_grad.flatten(order='F'),
                 Theta2_grad.flatten(order='F')]

    return J, grad
