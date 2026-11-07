import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import flexmm

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PublicApiTest(unittest.TestCase):
    def test_posterior_is_exposed_from_public_namespace(self):
        self.assertEqual(flexmm.Posterior.__module__, "flexmm.posterior")

    @unittest.skipUnless(HAS_TORCH, "torch is required for segmentation tests")
    def test_backend_kwargs_reach_feature_backend(self):
        from flexmm import SegmentationMap

        image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        segmentation = SegmentationMap(image)
        sentinel = object()
        with patch("flexmm.segmentation_map.seg._fit_model", return_value=sentinel) as fit:
            segmentation.fit_model(
                model="a",
                n_components=2,
                deepnet="vgg19",
                backend_kwargs={"pretrained": False, "device": "cpu"},
            )
        self.assertIs(segmentation.model_res["a"], sentinel)
        self.assertEqual(
            fit.call_args.kwargs["backend_kwargs"],
            {"pretrained": False, "device": "cpu"},
        )

    @unittest.skipUnless(HAS_TORCH, "torch is required for DINOv3 tests")
    def test_dinov3_requires_explicit_existing_weights(self):
        from flexmm.deepnets.dinov3 import load_model

        with self.assertRaisesRegex(ValueError, "explicit local weights path"):
            load_model(pretrained=True, weights=None)
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pth"
            with self.assertRaisesRegex(FileNotFoundError, str(missing)):
                load_model(pretrained=True, weights=missing)


if __name__ == "__main__":
    unittest.main()
