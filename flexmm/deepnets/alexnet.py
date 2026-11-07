import numpy as np
import torchvision.models as models


def get_n_list(ny, nx):
    return np.array([(int((ny - 11) / 4) + 2, int(((nx - 11) / 4) + 2))])


def build(ny, nx, *, device="cpu", pretrained=True, weights=None):
    if ny != 227 or nx != 227:
        raise ValueError("AlexNet FlexMM path expects a 227 x 227 input image")

    if weights is None:
        model = models.alexnet(pretrained=pretrained)
    else:
        model = models.alexnet(weights=weights)

    return {
        "model": model.features.to(device).eval(),
        "N_list": get_n_list(ny, nx),
        "d_list": np.array([64]),
        "neigh_size_list": 1.0 * np.array([17]),
        "default_layer": 1,
        "feature_kind": "conv2d",
    }
