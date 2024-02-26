from dataclasses import dataclass

import comet_ml
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import (
    compute_topk_accuracy,
    AvgMeterLossTopk,
    TqdmLossTopK,
)
from model import ClassificationBaseModel, get_device


@dataclass
class TrainConfig:
    grad_accum_interval: int
    log_interval_steps: int


@dataclass
class TrainOutput:
    loss: float
    top1: float
    train_step: int


def train(
    model: ClassificationBaseModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    current_train_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
    train_config: TrainConfig
) -> TrainOutput:
    """training loop for one epoch

    Args:
        model (ClassificationBaseModel): CNN model
        optimizer (Optimizer): optimizer
        scheduler (LRScheduler): learning rate (lr) scheduler
        loader (DataLoader): training dataset loader
        current_train_step (int): current step for training
        current_epoch (int): current epoch
        logger (comet_ml.Experiment): comet logger
        train_config (TrainInfo): information for training

    Returns:
        TrainOutput: train loss, train top1, steps for training
    """

    train_meters = AvgMeterLossTopk("train")

    model.train()
    device = get_device(model)

    with TqdmLossTopK(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            leave=False,
            unit='step',
    ) as progress_bar_step:
        progress_bar_step.set_description("[train    ]")

        for batch_index, batch in progress_bar_step:

            data, labels = batch  # (BCHW, B) for images or (BCTHW, B) for videos

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            if (
                train_config.grad_accum_interval == 1
                or batch_index % train_config.grad_accum_interval == 1
            ):
                optimizer.zero_grad()

            outputs = model(data, labels=labels)
            loss = outputs.loss
            loss.backward()

            train_topk = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            train_meters.update(loss, train_topk, batch_size)

            if current_train_step % train_config.log_interval_steps == 0:
                progress_bar_step.set_postfix_str_loss_topk(
                    current_train_step, loss, train_topk
                )
                logger.log_metrics(
                    train_meters.get_step_metrics_dict(),
                    step=current_train_step,
                    epoch=current_epoch,
                )

            if batch_index % train_config.grad_accum_interval == 0:
                optimizer.step()
                current_train_step += 1

    scheduler.step()

    logger.log_metrics(
        train_meters.get_epoch_metrics_dict(),
        step=current_train_step,
        epoch=current_epoch,
    )

    return TrainOutput(
        loss=train_meters.loss_meter.avg,
        top1=train_meters.topk_meters[0].avg,  # top1: topk[0] should be 1
        train_step=current_train_step
    )
