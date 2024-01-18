import os
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)


@dataclass
class ModelInfo:
    model_name: str
    use_pretrained: bool
    torch_home: str
    n_classes: int
    device: torch.device
    gpu_strategy: str


def model_factory(
        model_info: ModelInfo
) -> torch.nn:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        torch.nn: CNN model
    """

    if model_info.use_pretrained:
        # Specity the directory where a pre-trained model is stored.
        # Otherwise, by default, models are stored in users home dir `~/.torch`
        os.environ["TORCH_HOME"] = model_info.torch_home

    if model_info.model == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if model_info.use_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, model_info.n_classes)

    elif model_info.model == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if model_info.use_pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, model_info.n_classes)

    elif model_info.model == "x3d":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=model_info.use_pretrained,
            head_activation=None,  # removing nn.Softmax
        )
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Linear(in_features, model_info.n_classes)

    else:
        raise ValueError("invalid model_info.model")

    model = model.to(model_info.device)
    if model_info.gpu_strategy == "dp":
        model = nn.DataParallel(model)

    return model
