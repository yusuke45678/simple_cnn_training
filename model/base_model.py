from typing import Optional, Any, NamedTuple

import torch
from torch import nn


from model import ModelConfig


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ClassificationBaseModel(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.criterion = nn.CrossEntropyLoss()

        self.model = self.__create_dummy_model()

    def __create_dummy_model(self) -> nn.Module:
        return nn.Module()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:

        logits = self.model(pixel_values)

        if labels is None:
            return ModelOutput(
                logits=logits
            )

        loss = self.criterion(logits, labels)

        return ModelOutput(
            logits=logits,
            loss=loss
        )


def get_device(model: ClassificationBaseModel | nn.DataParallel | nn.Module) -> torch.device | Any:
    """get acutual device
    taken from https://github.com/pytorch/pytorch/issues/7460

    Returns:
        torch.device: device on which the model is loaded
    """

    if isinstance(model, nn.DataParallel):
        return next(model.module.parameters()).device

    return next(model.parameters()).device


def get_innermodel(model: ClassificationBaseModel | nn.DataParallel) -> nn.Module | Any:
    """get inner model of ClassificationBaseModel

    Args:
        model (ClassificationBaseModel | nn.DataParallel):
            a ClassificationBaseModel instance (or dp of it)

    Returns:
        nn.Module: inner model
    """
    if isinstance(model, nn.DataParallel):
        return model.module.model

    return model.model


def set_innermodel(
    model: ClassificationBaseModel,
    innter_model: nn.Module
) -> ClassificationBaseModel:
    """set inner model of ClassificationBaseModel

    Args:
        model (ClassificationBaseModel): a ClassificationBaseModel instance
        innter_model (nn.Module): an nn.Module instance

    Returns:
        ClassificationBaseModel: the ClassificationBaseModel instance
    """
    model.model = innter_model
    return model
