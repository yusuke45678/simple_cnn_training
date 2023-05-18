import os

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)


def model_factory(args, n_classes):
    """model factory

    Args:
        args (argparse): args
        n_classes (int): number of classes

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        torch.nn: CNN model
    """

    if args.use_pretrained:
        # Specity the directory where a pre-trained model is stored.
        # Otherwise, by default, models are stored in users home dir `~/.torch`
        os.environ["TORCH_HOME"] = args.torch_home

    if args.model == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if args.use_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif args.model == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if args.use_pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif args.model == "x3d":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=args.use_pretrained,
            head_activation=None,  # default is nn.Softmax, which is not for training
        )
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Linear(in_features, n_classes)

    else:
        raise ValueError("invalid args.model")

    return model
