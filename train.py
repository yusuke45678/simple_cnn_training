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

    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()

    model.eval()

    with torch.no_grad(), \
            tqdm(loader, total=len(loader), leave=False) as pbar_loss:

        pbar_loss.set_description('[val]')
        for data, labels in pbar_loss:

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
                    val_loss.val, val_loss.avg,
                    val_top1.val, val_top1.avg,
                    val_top5.val, val_top5.avg,
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

    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    model.train()

    with tqdm(
        enumerate(loader),
        total=len(loader),
        leave=False
    ) as pbar_loss:

        pbar_loss.set_description('[train]')
        for batch_index, (data, labels) in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            if batch_index % args.grad_accum == 0:
                optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()

            if (batch_index + 1) % args.grad_accum == 0:
                optimizer.step()
                global_steps += 1

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
                        train_loss.val, train_loss.avg,
                        train_top1.val, train_top1.avg,
                        train_top5.val, train_top5.avg,
                    ))

                experiment.log_metric(
                    'train_batch_loss', train_loss.val, step=global_steps)
                experiment.log_metric(
                    'train_batch_top1', train_top1.val, step=global_steps)
                experiment.log_metric(
                    'train_batch_top5', train_top5.val, step=global_steps)


    experiment.log_metric(
        'train_loss', train_loss.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'train_top1', train_top1.avg, step=global_steps, epoch=epoch)
    experiment.log_metric(
        'train_top5', train_top5.avg, step=global_steps, epoch=epoch)

    return global_steps
