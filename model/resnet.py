

from torch import nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)

from model import ModelConfig, ClassificationBaseModel


class ResNet18(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        weights = ResNet18_Weights.IMAGENET1K_V1 if model_config.use_pretrained else None
        self.model = resnet18(weights=weights)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            model_config.n_classes
        )


class ResNet50(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        weights = ResNet50_Weights.IMAGENET1K_V1 if model_config.use_pretrained else None
        self.model = resnet50(weights=weights)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            model_config.n_classes
        )
