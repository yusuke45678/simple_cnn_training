import torch
from torch import nn
from model import ModelConfig, ClassificationBaseModel


class X3D(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=model_config.use_pretrained,
            head_activation=None,  # removing nn.Softmax
        )

        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(in_features, model_config.n_classes)
