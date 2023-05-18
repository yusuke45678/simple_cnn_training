"""
    test models
"""

import random
import torch
from model import model_factory


class Dummy:
    """dummy class"""

    def __init__(self):
        self.model = ''
        self.use_pretrained = True
        self.torch_home = './pretrained_models'


def test_resnet18():
    """test resnet18"""

    args = Dummy()
    args.model = 'resnet18'
    for use_pretrained in [True, False]:
        n_classes = random.randint(2, 30)
        args.use_pretrained = use_pretrained

        model = model_factory(args, n_classes)
        assert model.__class__.__name__ == 'ResNet'
        assert model.fc.out_features == n_classes

        batch_size = random.randint(2, 30)
        input = torch.rand(batch_size, 3, 224, 224)
        output = model(input)  # BCHW --> (B, n_classes)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == n_classes
        assert len(output.shape) == 2


def test_resnet50():
    """test resnet18"""

    args = Dummy()
