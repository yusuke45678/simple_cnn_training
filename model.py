import os

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
        os.environ['TORCH_HOME'] = args.torch_home

    if args.model == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if args.use_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(
            model.fc.in_features, n_classes)
    elif args.model == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if args.use_pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(
            model.fc.in_features, n_classes)
    else:
        raise ValueError("invalid args.model")

    return model
