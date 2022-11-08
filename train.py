
import torch
import torch.nn as nn
from comet_ml import Experiment
from tqdm import tqdm

from args import get_args
from dataset import dataset_facory
from model import model_factory
from optimizer import optimizer_factory, scheduler_factory


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def val(model, criterion, optimizer, loader, device, iters, epoch):

    model.eval()
    loss_list = []
    acc_list = []
    with torch.no_grad(), \
            tqdm(loader, leave=False) as pbar_loss:

        pbar_loss.set_description('[val]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            batch_loss = loss.item() / data.size(0)
            _, preds = output.max(1)
            batch_acc = preds.eq(labels).sum().item() / data.size(0)

            pbar_loss.set_postfix_str(
                'loss={:.05f}, acc={:.03f}'.format(batch_loss, batch_acc))

            loss_list.append(batch_loss)
            acc_list.append(batch_acc)

    val_loss = sum(loss_list) / len(loss_list)
    val_acc = sum(acc_list) / len(acc_list)


def train(
    model,
    criterion,
    optimizer,
    loader,
    device: torch.device,
    iters: int,
    epoch,
    args
) -> int:

    experiment = Experiment()  # use .comet.config
    experiment.log_parameters(vars(args))

    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    model.train()

    with tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[train]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1.update(top1, batch_size)
            train_top5.update(top5, batch_size)
            train_loss.update(loss, batch_size)

            pbar_loss.set_postfix_str(
                'loss={:.3f}({:.3f}), '
                'top1={:.3f}({:.3f}), '
                'top5={:.3f}({:.3f})'.format(
                    train_loss.val, train_loss.avg,
                    train_top1.val, train_top1.avg,
                    train_top5.val, train_top5.avg,
                ))

            experiment.log_metric(
                'train_batch_loss', train_loss.val, step=iters)
            experiment.log_metric(
                'train_batch_top1', train_top1.val, step=iters)
            experiment.log_metric(
                'train_batch_top5', train_top5.val, step=iters)

            iters += 1

        experiment.log_metric(
            'train_loss', train_loss.avg, step=epoch)
        experiment.log_metric(
            'train_top1', train_top1.avg, step=epoch)
        experiment.log_metric(
            'train_top5', train_top5.avg, step=epoch)

    return iters


def main() -> None:

    args = get_args()

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_factory(args, n_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(args, model)
    scheduler = scheduler_factory(args, optimizer)

    iters = 0

    with tqdm(range(args.num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(epoch))

            iters = train(
                model, criterion, optimizer, train_loader,
                device, iters, epoch, args)

            if epoch % args.val_epochs:
                val(model, criterion, optimizer, val_loader, device,
                    iters, epoch)

            if args.use_scheduler:
                scheduler.update()


if __name__ == '__main__':
    main()
