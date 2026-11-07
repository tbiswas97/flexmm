from pathlib import Path

import numpy as np
import torch

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

MODEL_TO_EMBED_DIM = {
    MODEL_DINOV3_VITS: 384,
    MODEL_DINOV3_VITSP: 384,
    MODEL_DINOV3_VITB: 768,
    MODEL_DINOV3_VITL: 1024,
    MODEL_DINOV3_VITHP: 1280,
    MODEL_DINOV3_VIT7B: 4096,
}

MODEL_NAME = MODEL_DINOV3_VITB
PATCH_SIZE = 16
EMBED_DIM = MODEL_TO_EMBED_DIM[MODEL_NAME]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL_CACHE = {}


def get_default_feature_layers(model_name=MODEL_NAME):
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    return [n_layers // 4 - 1, n_layers - 4, n_layers - 1]


def normalize_feature_layers(feature_layers=None, model_name=MODEL_NAME):
    if feature_layers is None:
        feature_layers = get_default_feature_layers(model_name)
    elif isinstance(feature_layers, (int, np.integer)):
        feature_layers = [int(feature_layers)]
    else:
        feature_layers = [int(layer) for layer in feature_layers]

    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    if len(feature_layers) == 0:
        raise ValueError("At least one DINOv3 feature layer is required")

    for layer in feature_layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"DINOv3 layer {layer} is outside valid range 0..{n_layers - 1}"
            )

    return feature_layers


def get_n_list(ny, nx, patch_size=PATCH_SIZE):
    if ny % patch_size != 0 or nx % patch_size != 0:
        raise ValueError(
            f"DINOv3 input shape {(ny, nx)} must be divisible by patch size {patch_size}"
        )
    return np.array([(ny // patch_size, nx // patch_size)])


def load_model(
    *,
    model_name=MODEL_NAME,
    weights=None,
    pretrained=True,
    device="cpu",
    **hub_kwargs,
):
    if pretrained and weights is None:
        raise ValueError(
            "DINOv3 requires an explicit local weights path when pretrained=True"
        )
    if weights is not None:
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        weights = str(weights)

    cache_key = (
        model_name,
        str(weights),
        pretrained,
        str(device),
        tuple(sorted(hub_kwargs.items())),
    )

    if cache_key not in _MODEL_CACHE:
        from dinov3.hub import backbones

        load_kwargs = {"pretrained": pretrained}
        if weights is not None:
            load_kwargs["weights"] = weights
        load_kwargs.update(hub_kwargs)

        try:
            factory = getattr(backbones, model_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported DINOv3 model: {model_name}") from exc
        model = factory(**load_kwargs)
        _MODEL_CACHE[cache_key] = model.to(device).eval()

    return _MODEL_CACHE[cache_key]


def extract_features(model, im_torch, *, feature_layers=None, norm=True):
    device = next(model.parameters()).device
    x = im_torch.to(device)

    model_name = getattr(model, "flexmm_dino_model_name", MODEL_NAME)
    if feature_layers is None:
        feature_layers = getattr(model, "flexmm_dino_feature_layers", None)
    feature_layers = normalize_feature_layers(feature_layers, model_name)

    if norm:
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std

    # Pass a list so DINOv3 treats these as explicit block indices.
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            x,
            n=feature_layers,
            reshape=True,
            norm=True,
        )

    return np.array([feat.detach().cpu().numpy()[0] for feat in feats], dtype=object)


def build(
    ny,
    nx,
    *,
    device="cpu",
    weights=None,
    model_name=MODEL_NAME,
    pretrained=True,
    patch_size=PATCH_SIZE,
    embed_dim=None,
    feature_layers=None,
    **hub_kwargs,
):
    if model_name not in MODEL_TO_NUM_LAYERS:
        raise ValueError(f"Unsupported DINOv3 model: {model_name}")

    feature_layers = normalize_feature_layers(feature_layers, model_name)

    if embed_dim is None:
        embed_dim = MODEL_TO_EMBED_DIM[model_name]

    model = load_model(
        model_name=model_name,
        weights=weights,
        pretrained=pretrained,
        device=device,
        **hub_kwargs,
    )

    model.flexmm_dino_model_name = model_name
    model.flexmm_dino_feature_layers = feature_layers

    N_list = get_n_list(ny, nx, patch_size=patch_size)

    return {
        "model": model,
        "N_list": np.repeat(N_list, len(feature_layers), axis=0),
        "d_list": np.repeat(embed_dim, len(feature_layers)),
        "neigh_size_list": 1.0 * np.array([3, 1, 1])[: len(feature_layers)],
        "default_layer": len(feature_layers),
        "feature_kind": "dinov3",
        "extract_features": extract_features,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "feature_layers": feature_layers,
        "num_layers": MODEL_TO_NUM_LAYERS[model_name],
    }
