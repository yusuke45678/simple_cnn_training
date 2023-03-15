import torch
import os
from comet_ml.integration.pytorch import log_model, load_model


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411

    Args:
        output (torch.tensor): model output of the batch
        target (torch.tensor): labels of the batch
        topk (tuple, optional): k for computing top-k accuracy. Defaults to (1,).

    Returns:
        int or List[int]: top-k accuracy
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

        return res if len(res) > 1 else res[0]


def save_to_checkpoint(
    args,
    epoch,
    global_steps,
    acc,
    model,
    optimizer,
    scheduler,
    experiment,
):
    """save checkpoint file

    Args:
        args (argparse): args
        epoch (int): epoch
        global_steps (int): global step counter
        acc (float): accuracy (0 to 100)
        model (torch.nn): CNN model
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler
        experiment (comet_ml.Experiment): comet logger
    """

    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        args.save_checkpoint_dir,
        f'epoch{epoch}_steps{global_steps}_acc{acc:.2f}.pt')

    checkpoint_dict = {
        'epoch': epoch,
        'global_steps': global_steps,
        'accuracy': acc,
        'model_state_dict': model.state_dict() if not args.use_dp else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint_dict, checkpoint_file)

    # see https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/#saving-a-model
    log_model(experiment, checkpoint_dict, model_name=args.model)


def load_from_checkpoint(
        args,
        model,
        optimizer,
        scheduler,
        device
):
    """load from checkpoint file

    Args:
        args (argparse): args
        model (torch.nn): CNN model
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler
        device(torch.device): GPU device
    """

    if args.resume_from_checkpoint.startswith("experiment:"):
        # see https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/#loading-a-model
        checkpoint = load_model(args.resume_from_checkpoint)
    else:
        assert os.path.exists(args.resume_from_checkpoint)
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

    epoch = checkpoint['epoch']
    global_steps = checkpoint['global_steps']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = checkpoint['scheduler_state_dict']
    if scheduler:
        scheduler.load_state_dict(scheduler)

    return epoch, global_steps, model, optimizer, scheduler
