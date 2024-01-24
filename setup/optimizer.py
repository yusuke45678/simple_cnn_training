from dataclasses import dataclass
from typing import Iterator

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim import SGD, Adam


@dataclass
class OptimizerConfig:
    model_params: Iterator[Parameter]
    optimizer_name: str
    lr: float
    weight_decay: float
    momentum: float = 0.9


def configure_optimizer(
        optimizer_info: OptimizerConfig
) -> Optimizer:
    """optimizer factory

    Args:
        optimizer_info (OptimizerInfo): information for optimizer

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        torch.optim: optimizer
    """

    if optimizer_info.optimizer_name == "SGD":
        return SGD(
            optimizer_info.model_params,
            lr=optimizer_info.lr,
            momentum=optimizer_info.momentum,
            weight_decay=optimizer_info.weight_decay,
        )

    if optimizer_info.optimizer_name == "Adam":
        return Adam(
            optimizer_info.model_params,
            lr=optimizer_info.lr,
            weight_decay=optimizer_info.weight_decay,
        )

    raise ValueError("invalid optimizer_name")
