"""Small image and array helpers required by the FlexMM inference path."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(image):
    """Convert an RGB-like array to luminance while ignoring extra channels."""
    return np.dot(np.asarray(image)[..., :3], [0.299, 0.587, 0.114])


def gauss2d(x, y, sigma):
    """Evaluate an unnormalized centered two-dimensional Gaussian."""
    return np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * sigma**2))


def gt_pca_cluster_centers(features, labels):
    """Return the mean feature vector for each distinct initialization label."""
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("features and labels must contain the same number of samples")
    unique_labels = np.unique(labels)
    centers = np.asarray([features[labels == label].mean(axis=0) for label in unique_labels])
    return centers, len(unique_labels)


def _bounds(shape, spec, size, center):
    if spec is not None:
        y0, y1 = spec["y"]
        x0, x1 = spec["x"]
        return slice(y0, y1), slice(x0, x1)
    height, width = shape
    target_height, target_width = size
    if target_height > height or target_width > width:
        raise ValueError(f"crop size {size} exceeds image shape {(height, width)}")
    if center:
        y0 = (height - target_height) // 2
        x0 = (width - target_width) // 2
    else:
        y0 = x0 = 0
    return slice(y0, y0 + target_height), slice(x0, x0 + target_width)


def crop_rgb(image, spec=None, size=(256, 256), center=True):
    """Crop an image whose spatial axes are the first two axes."""
    image = np.asarray(image)
    y_slice, x_slice = _bounds(image.shape[:2], spec, size, center)
    return image[y_slice, x_slice, ...]


def crop(array, spec=None, size=(256, 256), center=True):
    """Crop a 2-D map or a stack whose spatial axes are the final two axes."""
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("crop input must have at least two dimensions")
    y_slice, x_slice = _bounds(array.shape[-2:], spec, size, center)
    return array[..., y_slice, x_slice]


def disp(images, shape=(1, 1), scale=(1, 1), marker=None, np_coords=True):
    """Display one image or a collection of images and return figure and axes."""
    rows, columns = shape
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(scale[1] * 3 * columns, scale[0] * 3 * rows),
        squeeze=False,
    )
    collection = [images] if rows * columns == 1 else list(images)
    for axis, image in zip(axes.flat, collection, strict=False):
        axis.imshow(image, cmap="gray")
        axis.set_axis_off()
    if marker is not None and rows * columns == 1:
        markers = marker if isinstance(marker, list) else [marker]
        for point in markers:
            row, column = point if np_coords else point[::-1]
            axes[0, 0].scatter(column, row, marker="o", s=50)
    return figure, axes[0, 0] if rows * columns == 1 else axes.squeeze()
