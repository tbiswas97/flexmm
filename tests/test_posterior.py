import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flexmm import Posterior


def make_flexmm(t=3, height=2, width=3, k=2, features=4, seed=3):
    rng = np.random.default_rng(seed)
    weights = rng.random((t, height, width, k))
    weights /= weights.sum(axis=-1, keepdims=True)
    diagonal = rng.uniform(0.5, 1.5, size=(t, k, features))
    covars = np.zeros((t, k, features, features))
    indices = np.arange(features)
    covars[:, :, indices, indices] = diagonal
    return SimpleNamespace(
        weights_t=weights,
        means_t=rng.normal(size=(t, k, features)),
        covars_t=covars,
        degrees_t=rng.uniform(2, 8, size=(t, k)),
        data_pca=rng.normal(size=(height, width, features)),
        im=rng.random((height * 2, width * 2, 3)),
    )


class PosteriorTest(unittest.TestCase):
    def test_accepts_common_deepnet_grid_shapes(self):
        cases = {
            "vgg": (256, 256),
            "alexnet": (56, 56),
            "dinov3": (16, 24),
        }
        for name, (height, width) in cases.items():
            with self.subTest(deepnet=name):
                flexmm = make_flexmm(
                    t=2, height=height, width=width, k=2, features=3
                )
                posterior = Posterior(flexmm, name, feature_index=2, n_r=5)
                self.assertEqual((posterior.height, posterior.width), (height, width))
                self.assertEqual(
                    posterior.analytical_mean((height - 1, width - 1)).shape,
                    (2,),
                )

    def test_dynamic_shapes_and_mixture_identity(self):
        flexmm = make_flexmm(t=4, height=3, width=5, k=3, features=7)
        posterior = Posterior(flexmm, "sample", feature_index=6, n_r=25)
        component_means = posterior.analytical_mean((2, 4), mix=False)
        mixed_means = posterior.analytical_mean((2, 4))
        expected = np.sum(flexmm.weights_t[:, 2, 4] * component_means, axis=1)
        self.assertEqual(component_means.shape, (4, 3))
        np.testing.assert_allclose(mixed_means, expected)
        self.assertEqual(posterior.pointwise_posterior((2, 4)).shape, (25, 4))

    def test_map_and_frozen_outputs(self):
        flexmm = make_flexmm()
        posterior = Posterior(flexmm, "sample", feature_index=1)
        coord = (1, 2)
        labels = flexmm.weights_t[:, coord[0], coord[1]].argmax(axis=1)
        component_means = posterior.analytical_mean(coord, mix=False)
        map_means = posterior.analytical_mean(coord, mix=False, _map=True)
        np.testing.assert_allclose(map_means, component_means[np.arange(3), labels])
        frozen = posterior.analytical_mean(coord, freeze_params=True)
        expected = np.sum(flexmm.weights_t[:, coord[0], coord[1]] * component_means[-1], axis=1)
        np.testing.assert_allclose(frozen, expected)

    def test_transient_is_opt_in_and_does_not_mutate_state(self):
        posterior = Posterior(make_flexmm(), "sample", feature_index=2)
        before_vars = posterior.vars_t.copy()
        before_dofs = posterior.dofs_t.copy()
        ordinary_before = posterior.analytical_var((0, 0))
        posterior.analytical_var((0, 0), include_transient=True)
        np.testing.assert_array_equal(posterior.vars_t, before_vars)
        np.testing.assert_array_equal(posterior.dofs_t, before_dofs)
        np.testing.assert_allclose(posterior.analytical_var((0, 0)), ordinary_before)

    def test_sandbox_uses_grid_shape_and_pickles(self):
        flexmm = make_flexmm(t=2, height=3, width=4)
        with tempfile.TemporaryDirectory() as directory:
            filekey = Path(directory) / "result"
            posterior = Posterior(flexmm, filekey, feature_index=0)
            result = posterior.sandbox(mod="map", analysis_window=(2, 3))
            output_path = Path(f"{filekey}_neural_statistics.pkl")
            self.assertEqual(result["frs"].shape, (1, 2, 3, 2))
            self.assertEqual(result["vars"].shape, (1, 2, 3, 2))
            self.assertEqual(result["entropy"].shape, (2, 3, 4))
            self.assertEqual(result["segmap"].shape, (2, 3, 4))
            self.assertEqual(result["feature_input"].shape, (3, 4))
            self.assertTrue(output_path.exists())
            with output_path.open("rb") as handle:
                loaded = pickle.load(handle)
            self.assertEqual(loaded["feature_index"], 0)

    def test_dataframe_coordinates_metadata_and_entropy(self):
        posterior = Posterior(make_flexmm(t=3), "sample", feature_index=1)
        frame = posterior.get_pointwise_df(
            [(0, 0), (1, 2)], metadata={"subject": ["a", "b"], "condition": "test"}
        )
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame["subject"].tolist(), ["a"] * 3 + ["b"] * 3)
        self.assertEqual(frame["condition"].tolist(), ["test"] * 6)
        self.assertTrue(
            {
                "row",
                "col",
                "mu_mixed",
                "mu_MAP",
                "var_mixed",
                "var_MAP",
                "entropy",
            }.issubset(frame.columns)
        )

    def test_validation_errors_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "keep=True"):
            Posterior(SimpleNamespace(), "sample", feature_index=0)
        flexmm = make_flexmm(features=2)
        with self.assertRaisesRegex(IndexError, "valid range"):
            Posterior(flexmm, "sample", feature_index=2)
        posterior = Posterior(flexmm, "sample", feature_index=1)
        with self.assertRaisesRegex(IndexError, "outside grid"):
            posterior.analytical_mean((10, 0))
        with self.assertRaisesRegex(ValueError, "exceeds FlexMM grid"):
            posterior.sandbox(pickle=False, analysis_window=10)


if __name__ == "__main__":
    unittest.main()
