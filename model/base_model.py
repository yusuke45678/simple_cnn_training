from dataclasses import dataclass

from typing import Optional, Iterator
try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from model import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class BaseModel():

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self.__create_dummy_model()

    def __create_dummy_model(self) -> nn.Module:
        return nn.Module()

    def train(self) -> Self:
        self.model.train()
        return self

    def eval(self) -> Self:
        self.model.eval()
        return self

    def to(self, device: torch.device) -> Self:
        self.model.to(device)
        self.model_config.device = device
        return self

    def get_device(self) -> torch.device:
        """get acutual device

        Returns:
            torch.device: device on which the model is loaded
        """
        # taken from https://github.com/pytorch/pytorch/issues/7460
        return next(self.model.parameters()).device

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        return self.model.named_modules(memo=None, prefix='', remove_duplicate=True)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def set_data_parallel(self) -> Self:
        self.model = nn.DataParallel(self.model)
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        return self.model

    def set_model(self, model: nn.Module) -> None:
        self.model = model


class ClassificationBaseModel(BaseModel):

    def __init__(self, model_info: ModelConfig):
        super().__init__(model_info)

        self.criterion = nn.CrossEntropyLoss()

    def __call__(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        logits = self.model(pixel_values)

        if labels is None:
            return ModelOutput(logits=logits)

        loss = self.criterion(logits, labels)

        return ModelOutput(
            logits=logits,
            loss=loss
        )
