import numpy as np

from ex2.sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

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
    z1 = X
    a1 = np.c_[np.ones(m), z1]

    z2 = np.dot(a1, Theta1.T)
    a2 = np.c_[np.ones(len(z2)), sigmoid(z2)]

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    p = np.argmax(a3, axis=1)

    # =========================================================================

    return p + 1  # add 1 to offset index of maximum in A row
