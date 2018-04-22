def linearKernel(x1, x2):
    """returns a linear kernel between x1 and x2
    and returns the value in sim
    """

    # Ensure that x1 and x2 are column vectors
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # Compute the kernel
    sim = x2.T @ x1  # dot product

    return sim
