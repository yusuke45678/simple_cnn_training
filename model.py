import os

import torch.nn as nn
from torchvision.models import resnet18


def model_factory(args, n_classes):

    if args.pretrain:
        # Specity the directory where a pre-trained model is stored.
        # Otherwise, by default, models are stored in users home dir `~/.torch`
        os.environ['TORCH_HOME'] = args.torch_home

    model = resnet18(pretrained=args.pretrain)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

    return model
