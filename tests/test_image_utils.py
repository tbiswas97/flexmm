import unittest

import numpy as np

from flexmm.image_utils import normalize_image
from flexmm.toolbox import crop, crop_rgb, gt_pca_cluster_centers


class NormalizeImageTest(unittest.TestCase):
    def test_normalizes_rgb_image(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        result = normalize_image(image)
        self.assertEqual(result.shape, (2, 4, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)

    def test_expands_grayscale_and_drops_alpha(self):
        grayscale = np.arange(6).reshape(2, 3)
        rgba = np.zeros((2, 3, 4))
        self.assertEqual(normalize_image(grayscale).shape, (2, 3, 3))
        self.assertEqual(normalize_image(rgba).shape, (2, 3, 3))

    def test_constant_image_becomes_zero(self):
        result = normalize_image(np.full((2, 3, 3), 7))
        np.testing.assert_array_equal(result, np.zeros((2, 3, 3), dtype=np.float32))

    def test_rejects_invalid_images(self):
        invalid_images = [
            np.array([]),
            np.zeros((2, 3, 2)),
            np.array([[[np.nan, 0, 1]]]),
            np.array([["not", "numeric"]]),
        ]
        for image in invalid_images:
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    normalize_image(image)

    def test_crops_image_and_map_stack_on_their_spatial_axes(self):
        image = np.arange(6 * 8 * 3).reshape(6, 8, 3)
        maps = np.arange(2 * 6 * 8).reshape(2, 6, 8)
        self.assertEqual(crop_rgb(image, size=(4, 4)).shape, (4, 4, 3))
        self.assertEqual(crop(maps, size=(4, 4)).shape, (2, 4, 4))

    def test_cluster_centers_include_zero_valued_features(self):
        features = np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 12.0]])
        centers, count = gt_pca_cluster_centers(features, np.array([0, 0, 1]))
        np.testing.assert_allclose(centers, [[1.0, 3.0], [10.0, 12.0]])
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
