# FlexMM

This work is an implementation of:

_Vacher, J., Launay, C. & Coen-Cagli, R. Flexibly Regularized Mixture Models and Application to Image Segmentation. Neural Networks 149, 107–123 (2022)._

With applications to: 

_Biswas, T., Vacher J., Molholm S., Mamassian P., & Coen-Cagli, R. Natural scene segmentation dynamics reveal iterative bayesian inference. bioRxiv (2026)_

and:

_Biswas, T. & Coen-Cagli, R. A probabilistic model of visual segmentation explains early visual cortical dynamics. NeurIPS (2026)_

Segments an image after extracting features from
VGG-19, AlexNet, or DINOv3 (DINOv3 weights must be pre-installed). 

Posterior inference (in `flexmm/posterior.py`) generates neural predictions. 

## Installation

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```


## Quickstart with VGG-19


import numpy as np
from flexmm import SegmentationMap
from PIL import Image 

image = np.load("examples/example-elk-im-arr.npy")

#an initial condition with quality segmentation is provided
initial_condition = np.load("examples/example-elk-prior.npy")

segmentation = SegmentationMap(image, image_id="example")
segmentation.fit_model(
    n_components=5, #user provided
    keep=True, #retains dynamics
    deepnet="vgg19", #options are vgg19, alexnet, dinov3
)
```

The final segmentation is saved as `segmentation.segmap`. 

`Posterior` requires `keep=True`. 

```
posterior = Posterior(
    segmentation,
    filekey=Path("outputs/example"),
    feature_index=0,
)
means = posterior.analytical_mean((0, 0))
```

## DINOv3 backend

DINOv3 (Sim&eacute;oni, 2025) is a state-of-the-art vision transformer.
To use, please pass an explicit local path after downloading weights from Meta AI

FlexMM with DINOv3 outputs outputs an (h//16, w//16) array for (h,w) sized inputs,
due to the patch sizes used in transformers, i.e. to get a (h,w) sized output, 
the input must be expanded up to (h*16,w*16). When patch sizes are a neighborhood of 
pixels, segmentation may be qualitatively improved with `spatial_smoothing=False`. 

```python
segmentation.fit_model(
    n_components=4,
    keep=True,
    layer_stop=3, #e.g. use the 3rd entry in feature_layers below
    deepnet="dinov3",
    backend_kwargs={
        "weights": "/absolute/path/to/dinov3_vitb16_weights.pth",
        "model_name": "dinov3_vitb16",
        "feature_layers": [2, 8, 11], #early, middle, and late layers
        "device": "cpu",
    },
    spatial_smoothing=False
)
```

## AlexNet backend

For AlexNet (Krizhevsky, 2012), use `deepnet="alexnet"`. 
AlexNet requires a 227 by 227 input. 
As of 11/2026, only the first layer of AlexNet has been tested for feature extraction.

## Development

```bash
python -m unittest discover -s tests -v
ruff check flexmm tests
python -m build
twine check dist/*
```

Tests use generated arrays and do not require example
images, downloaded weights, or network access.

## Licensing and attribution

The vendored `dinov3/` inference subset is derived from Meta's DINOv3 project
and remains governed by the [DINOv3 License](LICENSE.md). Some FlexMM
mixture-model source files contain GPLv3 provenance notices. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.
