import torch.optim as optim
from torch.optim import lr_scheduler


def optimizer_factory(args, model):
    """optimizer factory

    Args:
        args (argparse): args
        model (torch.nn): CNN model

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        torch.optim: optimizer
    """

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=args.betas,
            weight_decay=args.weight_decay)
    else:
        raise ValueError("invalid args.optimizer")

    return optimizer


def scheduler_factory(args, optimizer):
    """scheduler factory for learning rate

    Args:
        args (argparse): args
        optimizer (torch.optim): optimizer

    Returns:
        torch.optim.lr_scheduler: learning rate scheduler
    """
    if args.use_scheduler:
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    return scheduler
