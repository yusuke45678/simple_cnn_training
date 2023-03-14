import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy



def val(
    model,
    criterion,
    loader,
    device,
    global_steps: int,
    epoch: int,
    experiment
):
    """training loop for one epoch

    Args:
        model(torch.nn): CNN model
        criterion(torch.nn loss): loss function
        loader(torch.utils.data.DataLoader): validation dataset loader
        device(torch.device): GPU device
        global_steps(int): current step from the beginning
        epoch(int): current epoch
        experiment(comet_ml.Experiment): comet logger
    """

    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()

    model.eval()

    with torch.no_grad(), \
            tqdm(loader, total=len(loader), leave=False) as pbar_loss:

        pbar_loss.set_description('[val]')
        for data, labels in pbar_loss:

            data = data.to(device)  # BCHW
            labels = labels.to(device)  # B
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

    experiment.log_metric(
        'val_loss', val_loss.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'val_top1', val_top1.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'val_top5', val_top5.avg, step=global_steps, epoch=epoch)

    return


def train(
    model,
    criterion,
    optimizer,
    loader,
    device,
    global_steps: int,
    epoch: int,
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
        global_steps (int): current step from the beginning
        epoch (int): current epoch
        experiment (comet_ml.Experiment): comet logger
        args (argparse): args

    Returns:
        int: global_steps
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
        for batch_index, (data, labels) in pbar_loss:

            data = data.to(device)  # BCHW
            labels = labels.to(device)  # B
            batch_size = data.size(0)

            if batch_index % args.grad_accum == 0:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()

            if batch_index % args.grad_accum == 0:

                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                train_top1.update(top1, batch_size)
                train_top5.update(top5, batch_size)
                train_loss.update(loss, batch_size)

                pbar_loss.set_postfix_str(
                    'step={:d}, '
                    'loss={:6.4e}({:6.4e}), '
                    'top1={:6.2f}({:6.2f}), '
                    'top5={:6.2f}({:6.2f})'.format(
                        global_steps,
                        train_loss.value, train_loss.avg,
                        train_top1.value, train_top1.avg,
                        train_top5.value, train_top5.avg,
                    ))

                if global_steps % args.log_interval_steps == 0:
                    experiment.log_metric(
                        'train_batch_loss', train_loss.value, step=global_steps)
                    experiment.log_metric(
                        'train_batch_top1', train_top1.value, step=global_steps)
                    experiment.log_metric(
                        'train_batch_top5', train_top5.value, step=global_steps)

                optimizer.step()
                global_steps += 1


    experiment.log_metric(
        'train_loss', train_loss.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'train_top1', train_top1.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'train_top5', train_top5.avg, step=global_steps, epoch=epoch)

    return global_steps
