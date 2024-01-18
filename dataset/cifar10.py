from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def cifar10(args, train_transform, val_transform):

    train_dataset = CIFAR10(
        root=args.root,
        train=True,
        download=True,
        transform=train_transform)
    val_dataset = CIFAR10(
        root=args.root,
        train=False,
        download=True,
        transform=val_transform)
    n_classes = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    return train_loader, val_loader, n_classes
