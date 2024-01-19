from dataclasses import dataclass
import torch
from torch import nn

from typing import Optional
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from model import ModelInfo


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class BaseModel():

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.device = model_info.device
        self.model = nn.Module()  # dummy?

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: torch.device) -> Self:
        self.model.to(device)
        return self

    def get_parameter(self) -> nn.Parameter:
        return self.model.parameter()

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

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)

        self.criterion = nn.CrossEntropyLoss()

    def __call__(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        logits = self.model(pixel_values)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.criterion(logits, labels)

        return ModelOutput(
            logits=logits,
            loss=loss
        )
