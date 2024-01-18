import torch
from tqdm import tqdm
from typing import Tuple
import comet_ml

from utils import AverageMeter, accuracy


def val(
    model: torch.nn,
    criterion: torch.nn,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    global_step: int,
    current_epoch: int,
    experiment: comet_ml.Experiment,
) -> Tuple[float, float]:
    """validation for the current model

    Args:
        model(torch.nn): CNN model
        criterion(torch.nn loss): loss function
        loader(torch.utils.data.DataLoader): validation dataset loader
        device(torch.device): GPU device
        global_step(int): current step from the beginning
        current_epoch(int): current epoch
        experiment(comet_ml.Experiment): comet logger

    Returns:
        float: val loss
        float: val top1
    """

    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()

    model.eval()

    with torch.no_grad(), \
            tqdm(loader, total=len(loader), leave=False) as progress_bar_step:
        progress_bar_step.set_description("[val]")

        for batch in progress_bar_step:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            outputs = model(data)
            loss = criterion(outputs, labels)

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            val_top1.update(top1, batch_size)
            val_top5.update(top5, batch_size)
            val_loss.update(loss, batch_size)

            progress_bar_step.set_postfix_str(
                f"loss={val_loss.value:6.4e}({val_loss.avg:6.4e}), "
                f"top1={val_top1.value:6.2f}({val_top1.avg:6.2f}), "
                f"top5={val_top5.value:6.2f}({val_top5.avg:6.2f})"
            )

    experiment.log_metrics(
        {
            "val_loss_epoch": val_loss.avg,
            "val_top1_epoch": val_top1.avg,
            "val_top5_epoch": val_top5.avg,
        },
        step=global_step,
        epoch=current_epoch,
    )

    return val_loss.avg, val_top1.avg
