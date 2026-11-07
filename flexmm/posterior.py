"""Posterior summaries for a fitted FlexMM segmentation model.

The :class:`Posterior` class consumes the arrays exposed by
``flexmm.SegmentationMap`` after a ``keep=True`` fit.
It deliberately does not depend on an experiment-specific response object or
on the deep network used to produce the FlexMM features.
"""

from __future__ import annotations

import pickle as pickle_module
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import entropy


class Posterior:
    """Compute posterior statistics from the dynamic output of FlexMM.

    Parameters
    ----------
    flexmm : SegmentationMap-like object
        A fitted FlexMM object exposing ``weights_t``, ``means_t``,
        ``covars_t``, ``degrees_t``, and ``data_pca``. These dynamic arrays
        are produced by fitting model ``"c"`` with ``keep=True``.
    filekey : str or pathlib.Path
        Prefix used by :meth:`sandbox` when pickling its output.
    feature_index : int
        PCA/deep-feature dimension for which to compute the posterior.
    n_r : int, default=500
        Number of samples in the posterior-density grid.
    r_max : float, default=10
        Inclusive upper bound of the posterior-density grid.
    """

    _REQUIRED_OUTPUTS = (
        "weights_t",
        "means_t",
        "covars_t",
        "degrees_t",
        "data_pca",
    )

    def __init__(self, flexmm, filekey, feature_index, n_r=500, r_max=10):
        missing = [name for name in self._REQUIRED_OUTPUTS if not hasattr(flexmm, name)]
        if missing:
            raise ValueError(
                "FlexMM output is missing "
                + ", ".join(missing)
                + '; fit model="c" with keep=True before constructing Posterior'
            )
        if filekey is None or str(filekey) == "":
            raise ValueError("filekey must be a non-empty path prefix")
        if isinstance(feature_index, (bool, np.bool_)) or not isinstance(
            feature_index, (int, np.integer)
        ):
            raise TypeError("feature_index must be an integer")
        if isinstance(n_r, (bool, np.bool_)) or not isinstance(n_r, (int, np.integer)):
            raise TypeError("n_r must be an integer")
        if n_r < 2:
            raise ValueError("n_r must be at least 2")
        if not np.isfinite(r_max) or r_max <= 0:
            raise ValueError("r_max must be a finite positive number")

        self.flexmm = flexmm
        self.filekey = str(filekey)
        self.feature_index = int(feature_index)
        self.weights_t = np.asarray(flexmm.weights_t)
        self.all_means_t = np.asarray(flexmm.means_t)
        self.all_covars_t = np.asarray(flexmm.covars_t)
        self.dofs_t = np.asarray(flexmm.degrees_t)
        self.data_pca = np.asarray(flexmm.data_pca)
        self.image = getattr(flexmm, "im", None)
        self._validate_shapes()

        # Copies ensure opt-in transient calculations cannot modify the fitted
        # model or contaminate subsequent non-transient calculations.
        self.means_t = self.all_means_t[:, :, self.feature_index].copy()
        self.vars_t = self.all_covars_t[
            :, :, self.feature_index, self.feature_index
        ].copy()
        self.dofs_t = self.dofs_t.copy()
        self.n_r = int(n_r)
        self.r = np.linspace(0, float(r_max), self.n_r)
        self.sc = 1.0

    def _validate_shapes(self):
        if self.weights_t.ndim != 4:
            raise ValueError("weights_t must have shape (iterations, height, width, components)")
        if self.all_means_t.ndim != 3:
            raise ValueError("means_t must have shape (iterations, components, features)")
        if self.all_covars_t.ndim != 4:
            raise ValueError(
                "covars_t must have shape (iterations, components, features, features)"
            )
        if self.dofs_t.ndim != 2:
            raise ValueError("degrees_t must have shape (iterations, components)")
        if self.data_pca.ndim != 3:
            raise ValueError("data_pca must have shape (height, width, features)")

        self.T, self.height, self.width, self.K = self.weights_t.shape
        means_t, means_k, n_features = self.all_means_t.shape
        cov_t, cov_k, cov_d1, cov_d2 = self.all_covars_t.shape
        expected_tk = (self.T, self.K)
        if (means_t, means_k) != expected_tk:
            raise ValueError(
                f"means_t begins with {(means_t, means_k)}, expected {expected_tk}"
            )
        if (cov_t, cov_k) != expected_tk:
            raise ValueError(
                f"covars_t begins with {(cov_t, cov_k)}, expected {expected_tk}"
            )
        if self.dofs_t.shape != expected_tk:
            raise ValueError(f"degrees_t has shape {self.dofs_t.shape}, expected {expected_tk}")
        if (cov_d1, cov_d2) != (n_features, n_features):
            raise ValueError(
                "covars_t feature axes must be square and match the means_t feature axis"
            )
        if self.data_pca.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"data_pca spatial shape {self.data_pca.shape[:2]} does not match "
                f"weights_t spatial shape {(self.height, self.width)}"
            )
        if self.data_pca.shape[2] != n_features:
            raise ValueError(
                f"data_pca has {self.data_pca.shape[2]} features but means_t has {n_features}"
            )
        if not 0 <= self.feature_index < n_features:
            raise IndexError(
                f"feature_index {self.feature_index} is outside the valid range "
                f"0..{n_features - 1}"
            )

    def set_spike_count_multiplier(self, c):
        """Scale returned means by ``c`` and variances by ``c**2``."""
        if not np.isfinite(c):
            raise ValueError("spike count multiplier must be finite")
        self.sc = float(c)

    def _validate_coord(self, coord):
        if isinstance(coord, np.ndarray):
            coord = coord.tolist()
        if not isinstance(coord, Sequence) or isinstance(coord, (str, bytes)) or len(coord) != 2:
            raise TypeError("coord must be a (row, column) pair")
        row, col = coord
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (row, col)
        ):
            raise TypeError("coordinate values must be integers")
        row, col = int(row), int(col)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"coordinate {(row, col)} is outside grid shape {(self.height, self.width)}"
            )
        return row, col

    def _component_mean(self, coord, include_transient=False):
        row, col = self._validate_coord(coord)
        x = self.data_pca[row, col, self.feature_index]
        delta = x - self.means_t
        if include_transient:
            delta = delta.copy()
            delta[0] = self.data_pca[:, :, self.feature_index].mean()
        beta = self.dofs_t / 2
        gamma_term = gamma((self.dofs_t / 2) + 1) / gamma(
            (self.dofs_t / 2) + 0.5
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = ((beta / delta**2) + (1 / (2 * self.vars_t))) ** (-0.5)
        return out * gamma_term * self.sc

    def analytical_mean(
        self,
        coord,
        freeze_params=False,
        mix=True,
        _map=False,
        include_transient=False,
    ):
        """Return posterior means at a feature-grid coordinate."""
        row, col = self._validate_coord(coord)
        component_means = self._component_mean((row, col), include_transient)
        if not mix:
            if _map:
                labels = self.weights_t[:, row, col, :].argmax(axis=1)
                return component_means[np.arange(self.T), labels]
            return component_means
        weights = self.weights_t[:, row, col, :]
        if freeze_params:
            return np.sum(weights * component_means[-1], axis=1)
        return np.sum(weights * component_means, axis=1)

    def _component_var(self, coord, include_transient=False):
        row, col = self._validate_coord(coord)
        x = self.data_pca[row, col, self.feature_index]
        delta = x - self.means_t
        beta = self.dofs_t / 2
        variances = self.vars_t
        if include_transient:
            delta = delta.copy()
            beta = beta.copy()
            variances = variances.copy()
            delta[0] = self.data_pca[:, :, self.feature_index].mean()
            variances[0] = self.data_pca[:, :, self.feature_index].var()
            beta[0] = 1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            first_term = ((beta / delta**2) + (1 / (2 * variances))) ** (-1)
        gamma_term1 = gamma((self.dofs_t / 2) + 1.5) / gamma(
            (self.dofs_t / 2) + 0.5
        )
        gamma_term2 = (
            gamma((self.dofs_t / 2) + 1) / gamma((self.dofs_t / 2) + 0.5)
        ) ** 2
        return first_term * (gamma_term1 - gamma_term2) * self.sc**2

    def analytical_var(
        self,
        coord,
        freeze_params=False,
        mix=True,
        _map=False,
        include_transient=False,
    ):
        """Return posterior variances at a feature-grid coordinate."""
        row, col = self._validate_coord(coord)
        component_vars = self._component_var((row, col), include_transient)
        if not mix:
            if _map:
                labels = self.weights_t[:, row, col, :].argmax(axis=1)
                return component_vars[np.arange(self.T), labels]
            return component_vars
        component_means = self._component_mean((row, col), include_transient)
        weights = self.weights_t[:, row, col, :]
        if freeze_params:
            mean = np.sum(weights * component_means[-1], axis=1)
            second_moment = np.sum(
                weights * (component_vars[-1] + component_means[-1] ** 2), axis=1
            )
        else:
            mean = np.sum(weights * component_means, axis=1)
            second_moment = np.sum(
                weights * (component_vars + component_means**2), axis=1
            )
        return second_moment - mean**2

    # Retain the original private names for notebook compatibility. Feature
    # selection now belongs to the constructor.
    def _check_private_feature(self, pc):
        if pc is not None and int(pc) != self.feature_index:
            raise ValueError(
                "feature selection is fixed at construction; create another Posterior "
                "for a different feature_index"
            )

    def _analytical_mean(self, coord, pc=None, **kwargs):
        self._check_private_feature(pc)
        return self.analytical_mean(coord, **kwargs)

    def _analytical_var(self, coord, pc=None, **kwargs):
        self._check_private_feature(pc)
        return self.analytical_var(coord, **kwargs)

    def _pointwise_posterior_t(self, coord, t):
        row, col = self._validate_coord(coord)
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)):
            raise TypeError("t must be an integer")
        t = int(t)
        if not 0 <= t < self.T:
            raise IndexError(f"t {t} is outside the valid range 0..{self.T - 1}")
        x = self.data_pca[row, col, self.feature_index]
        return np.column_stack(
            [
                self._f_r_cond_x(
                    self.r,
                    x,
                    self.means_t[t, k],
                    self.vars_t[t, k],
                    self.dofs_t[t, k],
                )
                for k in range(self.K)
            ]
        )

    def pointwise_posterior(self, coord, freeze_params=False):
        """Return the mixed posterior density with shape ``(n_r, T)``."""
        row, col = self._validate_coord(coord)
        out = np.zeros((self.n_r, self.T))
        frozen = self._pointwise_posterior_t((row, col), self.T - 1)
        for t in range(self.T):
            component_density = frozen if freeze_params else self._pointwise_posterior_t((row, col), t)
            out[:, t] = component_density @ self.weights_t[t, row, col, :]
        return out

    def _coord_posterior(self, coord):
        return self.pointwise_posterior(coord)

    def _coord_posterior_t(self, coord, t):
        return self._pointwise_posterior_t(coord, t)

    @staticmethod
    def _f_r_cond_x(r, x, _mu, _var, _dof):
        g = np.asarray(r)
        alpha = _dof / 2
        beta = _dof / 2
        delta = x - _mu
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g_term = np.abs(g) ** (2 * alpha)
            exp_constant = (beta / np.abs(delta) ** 2) + (1 / (2 * _var))
            numerator = (
                exp_constant ** (alpha + 0.5)
                * g_term
                * np.exp(-(g**2) * exp_constant)
            )
        return numerator / gamma(alpha + 0.5)

    def _analysis_shape(self, analysis_window):
        if analysis_window is None:
            return self.height, self.width
        if isinstance(analysis_window, (int, np.integer)) and not isinstance(
            analysis_window, (bool, np.bool_)
        ):
            shape = (int(analysis_window), int(analysis_window))
        elif (
            isinstance(analysis_window, Sequence)
            and not isinstance(analysis_window, (str, bytes))
            and len(analysis_window) == 2
        ):
            shape = tuple(analysis_window)
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in shape
            ):
                raise TypeError("analysis_window dimensions must be integers")
            shape = tuple(int(value) for value in shape)
        else:
            raise TypeError("analysis_window must be None, an integer, or a (height, width) pair")
        if any(value <= 0 for value in shape):
            raise ValueError("analysis_window dimensions must be positive")
        if shape[0] > self.height or shape[1] > self.width:
            raise ValueError(
                f"analysis_window {shape} exceeds FlexMM grid {(self.height, self.width)}"
            )
        return shape

    def sandbox(
        self,
        pickle=True,
        mod=None,
        analysis_window=None,
        include_transient=False,
    ):
        """Build map-wide posterior summaries and optionally pickle them.

        ``mod`` may be ``None``/``"mixture"``, ``"map"``, or ``"freeze"``.
        The analysis window begins at the upper-left of the current FlexMM
        feature grid; by default the complete grid is analyzed.
        """
        modes = {None: "mixture", "mixture": "mixture", "map": "map", "freeze": "freeze"}
        if mod not in modes:
            raise ValueError("mod must be None, 'mixture', 'map', or 'freeze'")
        mode = modes[mod]
        height, width = self._analysis_shape(analysis_window)
        kwargs = {
            "mix": mode != "map",
            "_map": mode == "map",
            "freeze_params": mode == "freeze",
            "include_transient": include_transient,
        }
        means = np.empty((1, height, width, self.T))
        variances = np.empty_like(means)
        for row in range(height):
            for col in range(width):
                means[0, row, col] = self.analytical_mean((row, col), **kwargs)
                variances[0, row, col] = self.analytical_var((row, col), **kwargs)

        pmap = self.weights_t.copy()
        result = {
            "seg_means": self.all_means_t.copy(),
            "seg_vars": self.all_covars_t.copy(),
            "seg_dofs": self.dofs_t.copy(),
            "image": self.image,
            "entropy": entropy(pmap, axis=-1),
            "segmap": pmap.argmax(axis=-1),
            "frs": means,
            "vars": variances,
            "pmap": pmap,
            "feature_input": self.data_pca[:, :, self.feature_index].copy(),
            # Keep the historical key while making its new meaning explicit.
            "ff_input": self.data_pca[:, :, self.feature_index].copy(),
            "feature_index": self.feature_index,
        }
        if pickle:
            output_path = Path(f"{self.filekey}_neural_statistics.pkl")
            with output_path.open("wb") as handle:
                pickle_module.dump(result, handle)
        return result

    def get_pointwise_df(self, coords, metadata=None, include_transient=False):
        """Return mixed/MAP summaries for grid coordinates.

        Metadata values may be scalars, one value per coordinate, or one value
        per output row.
        """
        if isinstance(coords, np.ndarray):
            coords = coords.tolist()
        if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)):
            raise TypeError("coords must be a sequence of (row, column) pairs")
        coords = [self._validate_coord(coord) for coord in coords]
        n_points = len(coords)
        n_rows = n_points * self.T
        data = {
            "k": np.full(n_rows, self.K, dtype=int),
            "t": np.tile(np.arange(self.T), n_points),
            "point_idx": np.repeat(np.arange(n_points), self.T),
            "row": np.repeat([coord[0] for coord in coords], self.T),
            "col": np.repeat([coord[1] for coord in coords], self.T),
        }
        if n_points:
            data["mu_mixed"] = np.concatenate(
                [self.analytical_mean(coord, include_transient=include_transient) for coord in coords]
            )
            data["mu_MAP"] = np.concatenate(
                [
                    self.analytical_mean(
                        coord, mix=False, _map=True, include_transient=include_transient
                    )
                    for coord in coords
                ]
            )
            data["var_mixed"] = np.concatenate(
                [self.analytical_var(coord, include_transient=include_transient) for coord in coords]
            )
            data["var_MAP"] = np.concatenate(
                [
                    self.analytical_var(
                        coord, mix=False, _map=True, include_transient=include_transient
                    )
                    for coord in coords
                ]
            )
            entropy_map = entropy(self.weights_t, axis=-1)
            data["entropy"] = np.concatenate(
                [entropy_map[:, row, col] for row, col in coords]
            )
        else:
            for name in ("mu_mixed", "mu_MAP", "var_mixed", "var_MAP", "entropy"):
                data[name] = np.array([], dtype=float)

        frame = pd.DataFrame(data)
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping of column names to values")
            for name, value in metadata.items():
                if np.isscalar(value) or isinstance(value, (str, bytes)):
                    frame[name] = value
                    continue
                values = np.asarray(value)
                if values.ndim != 1:
                    raise ValueError(f"metadata column {name!r} must be one-dimensional")
                if len(values) == n_points:
                    frame[name] = np.repeat(values, self.T)
                elif len(values) == n_rows:
                    frame[name] = values
                else:
                    raise ValueError(
                        f"metadata column {name!r} has length {len(values)}; "
                        f"expected {n_points} or {n_rows}"
                    )
        return frame
