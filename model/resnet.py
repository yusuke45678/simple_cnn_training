

from torch import nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)

from model import ModelInfo, ClassificationBaseModel


class ResNet18(ClassificationBaseModel):

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)

        weights = ResNet18_Weights.IMAGENET1K_V1 if self.model_info.use_pretrained else None
        self.model = resnet18(weights=weights)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.model_info.n_classes
        )


class ResNet50(ClassificationBaseModel):

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)

        weights = ResNet50_Weights.IMAGENET1K_V1 if self.model_info.use_pretrained else None
        self.model = resnet50(weights=weights)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.model_info.n_classes
        )
