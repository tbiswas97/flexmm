# FlexMM

This work is an implementation of:

Vacher, J., Launay, C. & Coen-Cagli, R. Flexibly Regularized Mixture Models and Application to Image Segmentation. Neural Networks 149, 107–123 (2022).

With applications to: 

Biswas, T., Vacher J., Molholm S., Mamassian P., & Coen-Cagli, R. Natural scene segmentation dynamics reveal iterative bayesian inference. bioRxiv (2026)

and:

Biswas, T. & Coen-Cagli, R. A probabilistic model of visual segmentation explains early visual cortical dynamics. NeurIPS (2026)

FlexMM segments a caller-provided image by fitting flexible mixture models to
features from VGG-19, AlexNet, or DINOv3. 

This repository does not include datasets or model weights

## Installation

FlexMM requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development checks, install `.[dev]` instead.

## Quickstart with VGG-19

Load an image as a NumPy array with shape `(height, width)`, `(height, width,
3)`, or `(height, width, 4)`. FlexMM normalizes numeric images to `float32`
RGB in the range `[0, 1]`. VGG-19 is the default backend and the recommended
starting point; Torchvision downloads its pretrained weights on first use.

```python
from pathlib import Path

import matplotlib.image as mpimg

from flexmm import Posterior, SegmentationMap

image = mpimg.imread("/path/to/your/image.png")
segmentation = SegmentationMap(image, image_id="example")
segmentation.fit_model(
    model="c",
    n_components=4,
    keep=True,
    deepnet="vgg19",
)

posterior = Posterior(
    segmentation,
    filekey=Path("outputs/example"),
    feature_index=0,
)
means = posterior.analytical_mean((0, 0))
```

The final segmentation is available as `segmentation.segmap`. `Posterior`
requires a model `"c"` fit with `keep=True`. It validates the
iteration, spatial, component, and feature dimensions supplied by FlexMM and
works with backend-specific feature-grid sizes.

## DINOv3 backend

DINOv3 is optional at runtime but its weights must be downloaded separately
under Meta's terms. FlexMM never searches for or downloads DINOv3 weights.
Pass an explicit local path:

```python
segmentation.fit_model(
    model="c",
    n_components=4,
    keep=True,
    layer_stop=3,
    deepnet="dinov3",
    backend_kwargs={
        "weights": "/absolute/path/to/dinov3_vitb16_weights.pth",
        "model_name": "dinov3_vitb16",
        "feature_layers": [2, 8, 11],
        "device": "cpu",
    },
)
```

The image height and width must be divisible by the selected DINOv3 patch
size (16 by default). Supported aliases are `vgg19`, `alexnet`, and `dinov3`.
AlexNet requires a 227 by 227 input.

## Public API

- `SegmentationMap(image, image_id=0)` accepts only caller-provided images.
- `SegmentationMap.fit_model(...)` fits FlexMM and stores its dynamic outputs.
- `Posterior(fitted_model, filekey, feature_index, ...)` calculates posterior
  summaries and can serialize analysis output.

Backend-specific arguments belong in `backend_kwargs`; paths are never
hard-coded to a developer machine.

## Development

```bash
python -m unittest discover -s tests -v
ruff check flexmm tests
python -m build
twine check dist/*
```

The automated test suite uses generated arrays and does not require example
images, downloaded weights, or network access.

## Licensing and attribution

The vendored `dinov3/` inference subset is derived from Meta's DINOv3 project
and remains governed by the [DINOv3 License](LICENSE.md). Some FlexMM
mixture-model source files contain GPLv3 provenance notices. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution and
confirm that you have authority to release the complete combined work.

## Citation

Software citation metadata is provided in [CITATION.cff](CITATION.cff). Add
the accepted paper's final bibliographic entry and de-anonymized URL before
announcing the repository alongside the paper.
