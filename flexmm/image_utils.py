"""Image validation and normalization helpers for FlexMM."""

import numpy as np


def normalize_image(image):
    """Return a finite, normalized three-channel image array."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            "image must have shape (height, width), (height, width, 3), "
            f"or (height, width, 4); received {array.shape}"
        )
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("image must be a non-empty numeric array")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("image contains non-finite values")
    minimum = array.min()
    value_range = array.max() - minimum
    if value_range == 0:
        return np.zeros_like(array)
    return (array - minimum) / value_range
