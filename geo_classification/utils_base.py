from collections import OrderedDict
import logging
from pathlib import Path
from typing import Union
import torch
import torchvision


def build_base_model(arch: str, image_size=Union[None, int]):

    if "efficientnet" in arch:
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_pretrained(arch, image_size=image_size)
        nfeatures = model._fc.in_features
        model._dropout = torch.nn.Identity()
        model._fc = torch.nn.Identity()
    else:
        model = torchvision.models.__dict__[arch](pretrained=True)

        # get input dimension before classification layer
        if arch in ["mobilenet_v2"]:
            nfeatures = model.classifier[-1].in_features
            model = torch.nn.Sequential(*list(model.children())[:-1])
        elif arch in ["densenet121", "densenet161", "densenet169"]:
            nfeatures = model.classifier.in_features
            model = torch.nn.Sequential(*list(model.children())[:-1])
        elif "resne" in arch:
            # usually all ResNet variants
            nfeatures = model.fc.in_features
            model = torch.nn.Sequential(*list(model.children())[:-2])
        else:
            raise NotImplementedError

        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.flatten = torch.nn.Flatten(start_dim=1)
    return model, nfeatures


def load_weights_if_available(
    model: torch.nn.Module, classifier: torch.nn.Module, weights_path: Union[str, Path]
):

    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_classifier = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("classifier"):
            state_dict_classifier[k.replace("classifier.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model, classifier


def check_is_valid_torchvision_architecture(architecture: str):
    """Raises an ValueError if `architecture` is not part of `efficientnet-` or any available torchvision model"""

    if "efficientnet-" in architecture:
        return True

    available = sorted(
        name
        for name in torchvision.models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torchvision.models.__dict__[name])
    )
    if architecture not in available:
        raise ValueError(f"{architecture} not in {available}")

    return True
