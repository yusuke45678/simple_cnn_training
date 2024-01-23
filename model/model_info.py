from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    model_name: str = 'resnet18'
    use_pretrained: bool = True
    torch_home: str = './'
    n_classes: int = 10
    device: torch.device = torch.device('cuda')
    gpu_strategy: str = 'None'
