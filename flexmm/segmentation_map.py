from collections.abc import Mapping

import numpy as np

from flexmm import toolbox as tb
from flexmm.deepnets.registry import normalize_deepnet_name
from flexmm.image_utils import normalize_image
from flexmm.seg import segment as seg


def _non_integer_array(values):
    return np.asarray([value for value in values if not isinstance(value, int)])


class SegmentationMap:
    """Fit FlexMM segmentation to a caller-provided image.

    ``SegmentationMap(image)`` is the preferred interface. For compatibility
    with existing FlexMM callers, ``SegmentationMap((image_id, image),
    mode="array")`` is also accepted.
    """

    def __init__(self, image, mode=None, image_id=0):
        self.gs_im = None
        if mode not in (None, "array"):
            raise ValueError("only caller-provided images are supported")
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("legacy image input must be an (image_id, image) pair")
            image_id, image = image

        self.iid = str(image_id)
        self.iid_idx = image_id
        self.im = normalize_image(image)
        self._model_res = {}
        self.model_components = None
        self.seg_maps = {}
        self.cropped = False
        self.primary_seg_map = None
        self.rts = {}
        self.neigh_size_list = 1.0 * np.array(
            [17, 17, 13, 13, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3]
        )

    def __repr__(self) -> str:
        return (
            f"SegmentationMap(image_id={self.iid!r}, shape={self.im.shape}, "
            f"fitted={bool(self.model_res) if hasattr(self, 'model_res') else False}, "
            f"cropped={self.cropped})"
        )

    def make_grayscale(self, im):
        out = tb.rgb2gray(im)
        out = normalize_image(out)
        self.gs_im = out

        return out

    def fit_model(
        self,
        model="c",
        n_components=None,
        max_components=10,
        layer_start=0,
        layer_stop=16,
        layer_step=1,
        binning=False,
        use_crop=False,
        use_grayscale=False,
        keep=False,
        init=None,
        init_eps=None,
        spatial_smoothing=True,
        layer_normalization=True,
        reshape_deep_layers=True,
        deepnet="vgg19",
        backend_kwargs=None,
        n_pca=0.95,
    ):
        """
        Runs perceptual segmentation model on self.im
        Models are defined in models_deep_seg.py

        Parameters:
        -----------
        model : str
            string of options for which model to run, options are 'a','b','c', or combinations
            default behavior is to run model 'a' and model 'c'
        n_components : np.array
            Array of how many components for the model to return.
            If input is [4,5] the model will generate one result with 4 components, and one result with 5 components.
        max_components : int
            Maximum allowed number of components
        layer_start, layer_stop, layer_step : int
            Layers are selected according to
            ``[layer_start, layer_stop, layer_step]``.
        binning (depr) : bool
            Determines whether output segmentation maps at shallow layers are artificially downsampled (binned)
        use_crop : bool
            Determines whether to run the segmentation on the cropped image or the uncropped image
        use_grayscale : bool
            Determines whether to run the segmentation on a grayscale image or the original image
        keep : bool
            Set to True to keep the segmentation maps from every iteration of the EM algorithm
        init : np.array
            A 2D categorical label map used as the initial guess. It is
            automatically resized with nearest-neighbor sampling to match the
            first deep-network feature grid.
        init_eps: float
            This is the amount of uncertainty injected with the initial guess, if None 0.0001 is used as default
        spatial_smoothing : bool
            True if using spatial smoothing (2D Gaussian finite kernel)
            False if no spatial smoothing (2D Gaussian infinite kernel)
        deepnet : str
            determines which deep network is used for feature extraction, default is VGG19
        backend_kwargs : mapping, optional
            Backend-specific options such as ``weights``, ``model_name``,
            ``feature_layers``, and ``device``. DINOv3 weights must be supplied
            explicitly; FlexMM never downloads or discovers them implicitly.


        Raises:
        ------
        self.model_res : ndarray
            Model object defined in models_deep_seg.py
        self.seg_maps : dict
        self.flat_weights : ndarray
        self.weights_t : ndarray
        self.segmap : ndarray
        self.means_t : ndarray
        self.covars_t : ndarray
        self.degrees_t : ndarray
        self.responsibilites_t : ndarray
        self.likelihoods : ndarray
        self.flat_pca : ndarray
        self.data_pca : ndarray
        self.data : ndarray
        self.model_fitted: model Object
        """
        deepnet = normalize_deepnet_name(deepnet)
        if backend_kwargs is None:
            backend_kwargs = {}
        elif not isinstance(backend_kwargs, Mapping):
            raise TypeError("backend_kwargs must be a mapping")
        else:
            backend_kwargs = dict(backend_kwargs)

        if deepnet == "AlexNet":
            layer_stop = 1

        if deepnet == "dinov3":
            assert 1 <= layer_stop <= 3, "DINOv3 setup requires 1 <= layer_stop <= 3"

        if keep:
            assert model == "c", 'Must use model "c" if keep is True'

        if n_components is None:
            raise ValueError("n_components is required when fitting a provided image")
        n_components = np.atleast_1d(np.asarray(n_components, dtype=int))
        if n_components.ndim != 1 or len(n_components) == 0:
            raise ValueError("n_components must be an integer or a non-empty 1D sequence")
        if np.any(n_components <= 0):
            raise ValueError("n_components values must be positive")

        # if 3 not in n_components:
        # n_components = np.append(n_components, 3)


        # model_components is the number of components used by the model at runtime
        self.model_components = n_components

        if init is not None:
            # ensure that the initial guess is an array
            assert isinstance(init, np.ndarray)
            # init option only defined for model 'c'
            assert model == "c", 'Must use model "c" if init is not None'
            if init_eps is not None:
                k = self.model_components[-1]
                assert init_eps < (1 / 2) * (
                    1 / (k - 1)
                ), "Initialization epsilon value is too high for ground truth"

        if use_crop:
            if not self.cropped:
                raise ("Use crop method before calling with use_crop=True")
            else:
                model_im = self.c_im
        else:
            model_im = self.im

        if use_grayscale:
            model_im = self.make_grayscale(model_im)
            model_im = normalize_image(model_im)

        # Keep the feature-grid shape used by the fitted model so parse_layer
        # can reshape every saved output consistently. AlexNet's first
        # convolution maps its required 227 x 227 input to a 56 x 56 grid.
        self.deepnet = deepnet
        if deepnet == "AlexNet":
            self._model_output_shape = (56, 56)
        elif deepnet == "dinov3":
            h, w = model_im.shape[:2]
            patch_size = int(backend_kwargs.get("patch_size", 16))
            self._model_output_shape = (h // patch_size, w // patch_size)
        else:
            self._model_output_shape = model_im.shape[:2]

        # SEGMENTATION STEP:
        # calls files in seg/segment.py
        if keep:
            # run model 'c' keep results at each EM iteration
            # for layers until layer_stop
            if "c" in model:
                self._model_res["c"], self.__res_iter = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    keep=keep,
                    init=init,
                    init_eps=init_eps,
                    spatial_smoothing=spatial_smoothing,
                    neigh_size_list=self.neigh_size_list,
                    n_pca=n_pca,
                    deepnet=deepnet,
                    layer_normalization=layer_normalization,
                    reshape_deep_layers=reshape_deep_layers,
                    backend_kwargs=backend_kwargs,
                )

            # single_layer results here
            self.model_res = {}
            if layer_stop > 1:
                self._res_iter = self.__res_iter[:, layer_stop - 1, :, :]
                self.model_res["c"] = self._model_res["c"][layer_stop - 1, ...]
                # self.active_layer is initially layer_stop but is changed with parse_layer
                self.active_layer = layer_stop
                # self.layer_stop is a record of all layers that were fixed
                self.layer_stop = layer_stop
            else:
                self._res_iter = self.__res_iter
                self.model_res["c"] = self._model_res["c"]

            # parse attributes of the model
            reshape_size = self._model_output_shape

            weights = self._res_iter.T[0].squeeze()
            self.flat_weights = _non_integer_array(weights)
            self.weights_t = np.asarray(
                [
                    weight.reshape((*reshape_size, self.model_components[0]))
                    for weight in weights
                    if not isinstance(weight, int)
                ]
            )
            self.segmap = self.weights_t[-1, :, :, :].argmax(-1).astype("int")

            self.means_t = _non_integer_array(self._res_iter.T[1].squeeze())
            self.covars_t = _non_integer_array(self._res_iter.T[2].squeeze())
            self.degrees_t = _non_integer_array(self._res_iter.T[3].squeeze())

            responsibilities = self._res_iter.T[4].squeeze()
            self.responsibilities_t = np.asarray(
                [
                    resp.reshape((*reshape_size, self.model_components[0]))
                    for resp in responsibilities
                    if not isinstance(resp, int)
                ]
            )
            self.likelihoods = _non_integer_array(self._res_iter.T[5].squeeze())

            self.flat_pca = _non_integer_array(self._res_iter.T[6].squeeze()[0])
            self.data_pca = (
                self._res_iter.T[6].squeeze()[0].reshape((*reshape_size, -1))
            )

            self.data = self._res_iter.T[7].squeeze()[0].reshape((*reshape_size, -1))
            self.model_fitted = self.model_res["c"].squeeze()[2]

        else:
            # non-keep option only saves the last EM iteration
            # run model 'a'
            self.model_res = {}
            if "a" in model:
                self.model_res["a"] = seg._fit_model(
                    model_im,
                    model_type="a",
                    n_components=n_components,
                    layer=layer_stop,
                    deepnet=deepnet,
                    backend_kwargs=backend_kwargs,
                )
            # run model 'b'
            if "b" in model:
                self.model_res["b"] = seg._fit_model(
                    model_im,
                    model_type="b",
                    n_components=n_components,
                    layer=layer_stop,
                    deepnet=deepnet,
                    backend_kwargs=backend_kwargs,
                )
            # run model 'c'
            if "c" in model:
                self.model_res["c"] = seg._fit_model(
                    model_im,
                    model_type="c",
                    n_components=n_components,
                    layer=layer_stop,
                    init=init,
                    deepnet=deepnet,
                    backend_kwargs=backend_kwargs,
                )
        return None

    def parse_layer(self, layer):
        """
        Selects which layer in the model to parse output from
        Parsing output means reshaping data per layer

        Parameters:
        -----------
        layer : int
            layers are 1-indexed
        """
        assert layer > 0, "layers are 1-indexed"
        # set layer of interest here
        self._res_iter = self.__res_iter[:, layer - 1, :, :]
        self.model_res["c"] = self._model_res["c"][layer - 1, ...]
        # recalculate/reshape model outputs depending on layer
        reshape_size = getattr(self, "_model_output_shape", self.im.shape[:2])
        weights = self._res_iter.T[0].squeeze()
        self.flat_weights = _non_integer_array(weights)
        self.weights_t = np.asarray(
            [
                weight.reshape((*reshape_size, self.model_components[0]))
                for weight in weights
                if not isinstance(weight, int)
            ]
        )
        self.segmap = self.weights_t[-1, :, :, :].argmax(-1).astype("int")

        self.means_t = _non_integer_array(self._res_iter.T[1].squeeze())
        self.covars_t = _non_integer_array(self._res_iter.T[2].squeeze())
        self.degrees_t = _non_integer_array(self._res_iter.T[3].squeeze())

        responsibilities = self._res_iter.T[4].squeeze()
        self.responsibilities_t = np.asarray(
            [
                resp.reshape((*reshape_size, self.model_components[0]))
                for resp in responsibilities
                if not isinstance(resp, int)
            ]
        )
        self.likelihoods = _non_integer_array(self._res_iter.T[5].squeeze())

        self.flat_pca = _non_integer_array(self._res_iter.T[6].squeeze()[0])
        self.data_pca = (
            self._res_iter.T[6].squeeze()[0].reshape((*reshape_size, -1))
        )

        self.data = self._res_iter.T[7].squeeze()[0].reshape((*reshape_size, -1))
        self.model_fitted = self.model_res["c"].squeeze()[2]
        self.active_layer = layer

    def crop(
        self,
        spec={"y": (23, 278), "x": (23, 278)},
        size=(256, 256),
        center=True,
        RGB=True,
    ):
        """
        Crops an image, parameters specify different methods of cropping, based on toolbox.py -> crop

        In SegmentationMap object crops the following attributes:

        self.im -> self.c_im
        self.seg_map -> self.c_seg_map (nested dict of model results)

        Parameters:
        -----------
        spec : dict
            {y:(y1,y2),x:(x1,x2)}
        size : tup
            crop
        center : bool
            Determines whether size parameter is calculated from the center (True) or from the origin

        """
        if RGB:
            self.c_im = tb.crop_rgb(self.im, spec, size, center)
        else:
            self.c_im = tb.crop(self.im, spec, size, center)
        self.c_seg_maps = dict.fromkeys(self.seg_maps)
        d = self.c_seg_maps
        for key in d.keys():
            d[key] = dict.fromkeys(self.seg_maps[key])
            for _key in d[key].keys():
                to_crop = np.asarray(self.seg_maps[key][_key][:])
                d[key][_key] = np.asarray(tb.crop(to_crop, spec, size, center))

        return None

    def set_primary_seg_map(self, model="c", n_components=None, layer=0):
        """
        Sets the primary segmentation map for the segmentation map object

        Parameters:
        -----------
        model : str
            specifies which model to use as a key
        n_components : int
            number of components to use as a key
        layer : int
            layer INDEX to use as a key, index 0 means the 16th layer
            each proceeding index corresponds to 4 layers down

        Returns:
        --------
        None
        """
        if n_components is None:
            raise ValueError("n_components is required")
        if self.cropped:
            maps = self.c_seg_maps
        else:
            maps = self.seg_maps
        self.primary_seg_map = maps[model][n_components][layer]

    # Display functions: if self.cropped is True, then display the cropped image

    def disp(self, scale=(2, 2)):
        if self.cropped:
            im = self.c_im
        else:
            im = self.im
        tb.disp(im, scale=scale)

        return None

    def disp_seg_maps(self, model=None, n_components=None, layer=None):
        if self.cropped:
            maps = self.c_seg_maps
        else:
            maps = self.seg_maps

        if model is not None:
            if n_components is not None:
                if layer is not None:
                    tb.disp(maps[model][n_components][layer])
                else:
                    tb.disp(maps[model][n_components][:], shape=(2, 2))
        else:
            if self.primary_seg_map is not None:
                tb.disp(self.primary_seg_map)
            else:
                for key in maps.keys():
                    print(f"Model {key}")
                    for _key in maps[key].keys():
                        print(f"n_components: {_key}")
                        tb.disp(
                            maps[key][_key][:], shape=(2, 2)
                        )  # outputs maps at all 4 layers
