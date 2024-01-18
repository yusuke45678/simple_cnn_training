import torch
from torch.optim import lr_scheduler
from dataclasses import dataclass


@dataclass
class SchedulerInfo:
    optimizer: torch.optim
    use_scheduler: bool


def scheduler_factory(
        scheduler_info: SchedulerInfo
):
    """scheduler factory for learning rate

    Args:
        scheduler_info (SchedulerInfo): information for scheduler

    Returns:
        torch.optim.lr_scheduler: learning rate scheduler
    """
    if scheduler_info.use_scheduler:
        scheduler = lr_scheduler.StepLR(
            scheduler_info.optimizer,
            step_size=7,
            gamma=0.1
        )
    else:
        scheduler = None

    return scheduler
