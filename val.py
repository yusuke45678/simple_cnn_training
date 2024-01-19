import torch
from tqdm import tqdm
from typing import Tuple
import comet_ml
from dataclasses import dataclass

from utils import accuracy, AvgMeterLossTopk
from model import BaseModel


@dataclass
class ValidationOutput:
    loss: float
    top1: float


def validation(
    model: BaseModel,
    loader: torch.utils.data.DataLoader,
    global_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
) -> Tuple[float, float]:
    """validation for the current model

    Args:
        model(BaseModel): CNN model
        loader(torch.utils.data.DataLoader): validation dataset loader
        global_step(int): current step from the beginning
        current_epoch(int): current epoch
        logger(comet_ml.Experiment): comet logger

    Returns:
        ValidationOutput: val loss and val top1
    """

    val_meter = AvgMeterLossTopk("val")

    model.eval()
    device = model.device

    with torch.no_grad(), \
            tqdm(loader, total=len(loader), leave=False) as progress_bar_step:
        progress_bar_step.set_description("[val]")

        for batch in progress_bar_step:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            outputs = model(data, labels=labels)
            loss = outputs.loss

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            val_meter.update(loss, (top1, top5), batch_size)

            progress_bar_step.set_postfix_str(
                val_meter.get_set_postfix_str(global_step)
            )

    logger.log_metrics(
        val_meter.get_epoch_metrics_dict(),
        step=global_step,
        epoch=current_epoch,
    )

    return ValidationOutput(
        loss=val_meter.loss_meter.avg,
        top1=val_meter.topk_meter[0].avg  # top1: topk[0] should be 1
    )
