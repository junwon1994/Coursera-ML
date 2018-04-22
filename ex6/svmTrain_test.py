import numpy as np


# Save the model
class _Model():
    def __init__(self, X, y, kernelFunction, b, alphas, w):
        self.X = X
        self.y = y
        self.kernelFunction = kernelFunction
        self.b = b
        self.alphas = alphas
        self.w = w


def svmTrain_test(X, y, C, kernelFunction, tol=1e-3, max_passes=5):
    """ trains an SVM classifier and returns trained model.

    X is the matrix of training examples.
    Each row is a training example, and the jth column hold the jth feature.

    y is a vector.
    Each element is 1 for positive examples or 0 for negative examples.

    C is the standard SVM regularization parameter.

    tol is a tolerance value.
    It is used for determining equality of floating point numbers.

    max_passes is the number of iterations.
    It controls over the dataset (without changes to alpha).
    """
    # Data parameters
    m = np.size(X, 0)

    # Map 0 to -1
    y[y == 0] = -1

    # Variables
    alphas = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0
    eta = 0
    L = 0
    H = 0

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)
    #
    # We have implemented optimized vectorized version of the Kernels here so
    # that the svm training will run faster.
    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the Linear Kernel
        # This is equvalent to computing the kernel on every pair of examples
        K = X @ X.T
    else:
        # Vectorized RBF Kernel
        # This is equvalent to computing the kernel on every pair of examples
        X2 = np.sum(X**2, axis=1)
        K = X2 + X2.T - 2 * X @ X.T
        K = kernelFunction(1, 0)**K

    # Train
    print('\nTraining ...')
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + sum(alphas * y * K[:, i]) - y[i]
            if (y[i] * E[i] < -tol and alphas[i] < C) or (y[i] * E[i] > tol
                                                          and alphas[i] > 0):
                j = int(np.ceil(m * np.random.uniform())) - 1
                while j == i:
                    j = int(np.ceil(m * np.random.uniform())) - 1

                # Calculate E[j]
                E[j] = b + sum(alphas * y * K[:, j]) - y[j]

                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # Compute L and H
                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                # Compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= (y[j] * (E[i] - E[j])) / eta

                # Clip
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                # Check if change in alpha is significant
                if abs(alphas[j] - alpha_j_old) < tol:
                    # continue to next i
                    # replace anyway
                    alphas[j] = alpha_j_old
                    continue

                # Determine value for alpha i
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                # Compute b1 and b2
                b1 = b - E[i] - y[i] * (
                    alphas[i] - alpha_i_old) * K[i, j] - y[j] * (
                        alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - y[i] * (
                    alphas[i] - alpha_i_old) * K[i, j] - y[j] * (
                        alphas[j] - alpha_j_old) * K[j, j]

                # Compute b
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        print('.', end='', flush=True)
        dots += 1
        if dots > 78:
            dots = 0
            print()

    print(' Done! \n')

    idx = alphas > 0
    model = _Model(X[idx, :], y[idx], kernelFunction, b, alphas[idx],
                   ((alphas * y).T @ X).T)

    return model
