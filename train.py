import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy
from typing import Tuple


def val(
    model,
    criterion,
    loader,
    device,
    global_step: int,
    current_epoch: int,
    experiment,
    args
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
            tqdm(loader, total=len(loader), leave=False) as pbar_loss:

        pbar_loss.set_description('[val]')
        for batch in pbar_loss:

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

            pbar_loss.set_postfix_str(
                'loss={:6.4e}({:6.4e}), '
                'top1={:6.2f}({:6.2f}), '
                'top5={:6.2f}({:6.2f})'.format(
                    val_loss.value, val_loss.avg,
                    val_top1.value, val_top1.avg,
                    val_top5.value, val_top5.avg,
                ))

    experiment.log_metrics(
        {
            'val_loss_epoch': val_loss.avg,
            'val_top1_epoch': val_top1.avg,
            'val_top5_epoch': val_top5.avg,
        },
        step=global_step, epoch=current_epoch)

    return val_loss.avg, val_top1.avg


def train(
    model,
    criterion,
    optimizer,
    loader,
    device,
    global_step: int,
    current_epoch: int,
    experiment,
    args
) -> int:
    """training loop for one epoch

    Args:
        model (torch.nn): CNN model
        criterion (torch.nn loss): loss function
        optimizer (torch.optim): optimizer
        loader (torch.utils.data.DataLoader): training dataset loader
        device (torch.device): GPU device
        global_step (int): current step from the beginning
        current_epoch (int): current epoch
        experiment (comet_ml.Experiment): comet logger
        args (argparse): args

    Returns:
        int: global_step
    """

    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    model.train()

    with tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        leave=False
    ) as pbar_loss:

        pbar_loss.set_description('[train]')
        for batch_index, batch in pbar_loss:

            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            if args.grad_accum == 1 \
                    or batch_index % args.grad_accum == 1:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1.update(top1, batch_size)
            train_top5.update(top5, batch_size)
            train_loss.update(loss, batch_size)

            if global_step % args.log_interval_steps == 0:
                pbar_loss.set_postfix_str(
                    'step={:d}, '
                    'loss={:6.4e}({:6.4e}), '
                    'top1={:6.2f}({:6.2f}), '
                    'top5={:6.2f}({:6.2f})'.format(
                        global_step,
                        train_loss.value, train_loss.avg,
                        train_top1.value, train_top1.avg,
                        train_top5.value, train_top5.avg,
                    ))
                experiment.log_metrics(
                    {
                        'train_loss_step': train_loss.value,
                        'train_top1_step': train_top1.value,
                        'train_top5_step': train_top5.value,
                    },
                    step=global_step, epoch=current_epoch)

            if batch_index % args.grad_accum == 0:
                optimizer.step()
                global_step += 1

    experiment.log_metrics(
        {
            'train_loss_epoch': train_loss.avg,
            'train_top1_epoch': train_top1.avg,
            'train_top5_epoch': train_top5.avg,
        },
        step=global_step, epoch=current_epoch)

    return global_step
