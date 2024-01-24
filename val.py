from dataclasses import dataclass
from tqdm import tqdm

import comet_ml
import torch
from torch.utils.data import DataLoader

from utils import accuracy, AvgMeterLossTopk
from model import BaseModel


@dataclass
class ValidationOutput:
    loss: float
    top1: float
    val_step: int


def validation(
    model: BaseModel,
    val_loader: DataLoader,
    current_val_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
) -> ValidationOutput:
    """validation for the current model

    Args:
        model(BaseModel): CNN model
        loader(DataLoader): validation dataset loader
        current_val_step(int): current step for validation
        current_epoch(int): current epoch
        logger(comet_ml.Experiment): comet logger

    Returns:
        ValidationOutput: val loss, val top1, steps for validation
    """

    val_meter = AvgMeterLossTopk("val")

    model.eval()

    with torch.no_grad(), tqdm(
        val_loader,
        total=len(val_loader),
        leave=False
    ) as progress_bar_step:
        progress_bar_step.set_description("[val]")

        for batch in progress_bar_step:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}

            data = data.to(model.get_device())
            labels = labels.to(model.get_device())
            batch_size = data.size(0)

            outputs = model(data, labels=labels)
            loss = outputs.loss

            top1, top5 = accuracy(outputs.logits, labels, topk=(1, 5))
            val_meter.update(loss, (top1, top5), batch_size)

            progress_bar_step.set_postfix_str(
                val_meter.get_postfix_str(current_val_step)
            )
            logger.log_metrics(
                val_meter.get_step_metrics_dict(),
                step=current_val_step,
                epoch=current_epoch,
            )
            current_val_step += 1

    logger.log_metrics(
        val_meter.get_epoch_metrics_dict(),
        step=current_val_step,
        epoch=current_epoch,
    )

    return ValidationOutput(
        loss=val_meter.loss_meter.avg,
        top1=val_meter.topk_meters[0].avg,  # top1: topk[0] should be 1
        val_step=current_val_step
    )
