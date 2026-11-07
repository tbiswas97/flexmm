import numpy as np
import torchvision.models as models

N_CHANNELS = np.array(
    [
        64,
        64,
        128,
        128,
        256,
        256,
        256,
        256,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ]
)

NEIGH_SIZE_LIST = 1.0 * np.array(
    [17, 17, 13, 13, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3]
)


def get_n_list(ny, nx):
    return np.array(
        [
            (ny, nx),
            (ny, nx),
            (ny // 2, nx // 2),
            (ny // 2, nx // 2),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 4, nx // 4),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 8, nx // 8),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
            (ny // 16, nx // 16),
        ]
    )


def build(ny, nx, *, device="cpu", pretrained=True, weights=None):
    if weights is None:
        model = models.vgg19(pretrained=pretrained)
    else:
        model = models.vgg19(weights=weights)

    return {
        "model": model.features.to(device).eval(),
        "N_list": get_n_list(ny, nx),
        "d_list": N_CHANNELS.copy(),
        "neigh_size_list": NEIGH_SIZE_LIST.copy(),
        "default_layer": 16,
        "feature_kind": "conv2d",
    }
