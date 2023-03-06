import torch.optim as optim
from torch.optim import lr_scheduler


def optimizer_factory(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr, betas=args.betas)
    else:
        raise ValueError("invalid args.optimizer")

    return optimizer


def scheduler_factory(args, optimizer):
    if args.use_scheduler:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    return scheduler
