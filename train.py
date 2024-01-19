import torch
from tqdm import tqdm
import comet_ml
from dataclasses import dataclass

from utils import accuracy, AvgMeterLossTopk
from model import BaseModel


@dataclass
class TrainInfo:
    grad_accum_interval: int
    log_interval_steps: int


def train_one_epoch(
    model: BaseModel,
    optimizer: torch.optim,
    loader: torch.utils.data.DataLoader,
    global_step: int,
    current_epoch: int,
    experiment: comet_ml.Experiment,
    train_info: TrainInfo
) -> int:
    """training loop for one epoch

    Args:
        model (BaseModel): CNN model
        optimizer (torch.optim): optimizer
        loader (torch.utils.data.DataLoader): training dataset loader
        global_step (int): current step from the beginning
        current_epoch (int): current epoch
        experiment (comet_ml.Experiment): comet logger
        train_info (TrainInfo): information for training

    Returns:
        int: global_step
    """

    train_meter = AvgMeterLossTopk("train")

    model.train()
    device = model.device

    with tqdm(enumerate(loader, start=1), total=len(loader), leave=False) as progress_bar_step:
        progress_bar_step.set_description("[train]")

        for batch_index, batch in progress_bar_step:

            data, labels = batch  # (BCHW, B) for images or (BCTHW, B) for videos

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            if (
                train_info.grad_accum_interval == 1
                or batch_index % train_info.grad_accum_interval == 1
            ):
                optimizer.zero_grad()

            outputs = model(data, labels=labels)
            loss = outputs.loss
            loss.backward()

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_meter.update(loss, (top1, top5), batch_size)

            if global_step % train_info.log_interval_steps == 0:
                progress_bar_step.set_postfix_str(
                    train_meter.get_set_postfix_str(global_step)
                )
                experiment.log_metrics(
                    train_meter.get_step_metrics_dict(),
                    step=global_step,
                    epoch=current_epoch,
                )

            if batch_index % train_info.grad_accum_interval == 0:
                optimizer.step()
                global_step += 1

    experiment.log_metrics(
        train_meter.get_epoch_metrics_dict(),
        step=global_step,
        epoch=current_epoch,
    )

    return global_step
