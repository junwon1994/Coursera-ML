import numpy as np
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Set example_width automatically if not passed in
    if not example_width:
        example_width = round(X.shape[1]**0.5)

    # Gray Image
    plt.set_cmap('gray')

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(m**0.5)
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones(
        (pad + display_rows * (example_height + pad), pad + display_cols *
         (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > (m - 1):
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex]))

            r = pad + j * (example_height + pad)
            c = pad + i * (example_width + pad)

            display_array[r:r + example_height, c:c + example_width] = X[
                curr_ex].reshape(
                    example_height, example_width, order='F') / max_val
            curr_ex += 1

        if curr_ex > (m - 1):
            break

    # Display Image
    plt.imshow(display_array)

    # Do not show axis
    plt.axis('off')

    plt.show(block=True)
