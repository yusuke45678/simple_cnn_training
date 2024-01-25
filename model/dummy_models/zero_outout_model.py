from typing import Optional

import torch

from model import ClassificationBaseModel, ModelOutput


class ZeroOutputModel(ClassificationBaseModel):

    def __call__(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        dummy_logits = torch.zeros(
            (
                batch_size,
                self.model_config.n_classes
            ),
            device=device
        )

        if labels is None:
            dummy_loss = None
        else:
            dummy_loss = torch.Tensor([0.0]).to(device)[0]

        return ModelOutput(
            logits=dummy_logits,
            loss=dummy_loss
        )
