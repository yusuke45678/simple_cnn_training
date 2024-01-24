import os
from typing import Tuple, Optional

import comet_ml
from comet_ml.integration.pytorch import log_model, load_model

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_to_checkpoint(
    save_checkpoint_dir: str,
    current_epoch: int,
    current_train_step: int,
    current_val_step: int,
    acc: float,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    experiment_logger: comet_ml.Experiment,
) -> dict:
    """save checkpoint to file

    Args:
        save_checkpoint_dir: path to dir where checkpoint is saved
        current_epoch (int): epoch
        current_train_step (int): global step counter for training
        current_val_step (int): global step counter for validation
        acc (float): accuracy (0 to 100)
        model (torch.nn): CNN model
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler
        experiment_logger (comet_ml.Experiment): comet logger
    """

    save_checkpoint_dir = os.path.join(
        save_checkpoint_dir,
        experiment_logger.project_name.replace(" ", "_"),
        experiment_logger.name.replace(" ", "_"),
    )
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        save_checkpoint_dir,
        f"epoch{current_epoch}_step{current_train_step}_acc={acc:.2f}.pt",
    )

    checkpoint_dict = {
        "current_epoch": current_epoch,
        "current_train_step": current_train_step,
        "current_val_step": current_val_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint_dict, checkpoint_file)

    return checkpoint_dict


def save_to_comet(
    checkpoint_dict: dict,
    log_model_name: str,
    experiment_logger: comet_ml.Experiment,
):
    """save checkpoint to comet

    Args:
        checkpoint_dict (dict): checkpoint dictionary
        log_model_name (str): logged model name
        experiment_logger (comet_ml.Experiment): comet logger
    """
    # see https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/#saving-a-model
    log_model(experiment_logger, checkpoint_dict, model_name=log_model_name)


def load_from_checkpoint(
    checkpoint_to_resume: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    device: torch.device
) -> Tuple[int, int, int, nn.Module, Optimizer, Optional[LRScheduler]]:
    """load from checkpoint file or online comet_ml

    Args:
        checkpoint_to_resume (str): path to checkpoint to resume.
            if it starts with "experiment:" then load from comet_ml instead of local file
        model (torch.nn): CNN model
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler
        device(torch.device): GPU device
    """

    if checkpoint_to_resume.startswith("experiment:"):
        # see https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/#loading-a-model
        checkpoint = load_model(checkpoint_to_resume)
    else:
        assert os.path.exists(checkpoint_to_resume)
        checkpoint = torch.load(checkpoint_to_resume, map_location=device)

    current_epoch: int = checkpoint["current_epoch"]
    current_train_step: int = checkpoint["current_train_step"]
    current_val_step: int = checkpoint["current_val_step"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # scheduler_dict = checkpoint["scheduler_state_dict"]
    # if scheduler_dict:
    #     scheduler.load_state_dict(scheduler_dict)
    if scheduler and "scheduler_state_dict" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        scheduler = None

    return current_epoch, current_train_step, current_val_step, model, optimizer, scheduler
