from dataclasses import dataclass
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR
)


@dataclass
class SchedulerConfig:
    optimizer: Optimizer
    use_scheduler: bool


def configure_scheduler(
        scheduler_info: SchedulerConfig
) -> Optional[LRScheduler]:
    """scheduler factory for learning rate

    Args:
        scheduler_info (SchedulerInfo): information for scheduler

    Returns:
        LRScheduler: learning rate scheduler
    """
    if scheduler_info.use_scheduler:
        return StepLR(
            scheduler_info.optimizer,
            step_size=10,  # every 10 epoch
            gamma=0.1  # lr = lr * 0.1
        )

    return None
