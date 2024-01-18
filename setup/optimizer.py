from dataclasses import dataclass
from typing import Iterator
import torch.optim as optim
import torch


@dataclass
class OptimizerInfo:
    optimizer_name: str
    lr: float
    weight_decay: float
    momentum: float = 0.9
    betas: float = 0.99
    model_params: Iterator[torch.nn.parameter.Parameter]


def optimizer_factory(
        optimizer_info: OptimizerInfo
):
    """optimizer factory

    Args:
        optimizer_info (OptimizerInfo): information for optimizer

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        torch.optim: optimizer
    """

    if optimizer_info.optimizer_name == "SGD":
        optimizer = optim.SGD(
            optimizer_info.model_parameters,
            lr=optimizer_info.lr,
            momentum=optimizer_info.momentum,
            weight_decay=optimizer_info.weight_decay,
        )

    elif optimizer_info.optimizer_name == "Adam":
        optimizer = optim.Adam(
            optimizer_info.model_parameters,
            lr=optimizer_info.lr,
            betas=optimizer_info.betas,
            weight_decay=optimizer_info.weight_decay,
        )

    else:
        raise ValueError("invalid args.optimizer")

    return optimizer
