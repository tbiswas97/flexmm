def normalize_deepnet_name(name):
    if name is None:
        return "vgg19"

    aliases = {
        "alexnet": "AlexNet",
        "AlexNet": "AlexNet",
        "vgg": "vgg19",
        "vgg19": "vgg19",
        "dino": "dinov3",
        "dinov3": "dinov3",
        "DINOV3": "dinov3",
        "dinov3_vits16": "dinov3",
    }

    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"Unknown deepnet: {name}") from exc


def get_deepnet(name, ny, nx, *, device="cpu", layer=None, **kwargs):
    name = normalize_deepnet_name(name)

    if name == "vgg19":
        from flexmm.deepnets import vgg19

        spec = vgg19.build(ny=ny, nx=nx, device=device, **kwargs)
    elif name == "AlexNet":
        from flexmm.deepnets import alexnet

        spec = alexnet.build(ny=ny, nx=nx, device=device, **kwargs)
    elif name == "dinov3":
        from flexmm.deepnets import dinov3

        spec = dinov3.build(ny=ny, nx=nx, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown deepnet: {name}")

    spec["name"] = name
    spec["layer"] = spec["default_layer"] if layer is None else layer
    return spec
