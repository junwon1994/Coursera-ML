import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Useful values
    m = len(X)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #

    # Input Layer
    z_1 = X
    a_1 = np.c_[np.ones(m), z_1]

    # Hidden Layer
    z_2 = a_1 @ Theta1.T
    a_2 = np.c_[np.ones(m), sigmoid(z_2)]

    # Output Layer
    z_3 = a_2 @ Theta2.T
    a_3 = sigmoid(z_3)

    H = a_3

    p = np.argmax(H, axis=1)

    if m == 1:
        p = p.squeeze()

    # =========================================================================

    return p + 1  # add 1 to offset index of maximum in A row
