# from torch import nn
from torchvision.models import (
    vgg19,
    VGG19_Weights,
)

from model import ModelConfig, ClassificationBaseModel


class VGG19(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()
        # self.replace_pretrained_head()

    def prepare_model(self):
        self.model = vgg19(
            weights=VGG19_Weights.DEFAULT
            if self.model_config.use_pretrained else None)

    # def replace_pretrained_head(self):
    #    self.model.head = nn.Linear(
    #        self.model.head.in_features,
    #        self.model_config.n_classes
    #    )
