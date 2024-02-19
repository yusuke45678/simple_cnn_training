from typing import Literal
from dataclasses import dataclass

import torch

SupportedModels = Literal["resnet18", "resnet50", "abn_r50", "vit_b", "x3d", "zero_output_dummy"]


@dataclass
class ModelConfig:
    model_name: SupportedModels = "resnet18"
    use_pretrained: bool = True
    torch_home: str = "./"
    n_classes: int = 10
    device: torch.device = torch.device("cuda")
    gpu_strategy: str = "None"
