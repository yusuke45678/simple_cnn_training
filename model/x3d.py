import torch
from torch import nn
from model import ModelInfo, ClassificationBaseModel


class X3D(ClassificationBaseModel):

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=model_info.use_pretrained,
            head_activation=None,  # removing nn.Softmax
        )

        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(in_features, model_info.n_classes)
