from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
import os


def transform_factory(args, do_crop=True):
    if do_crop:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])
        ])
    return transform


def dataset_facory(args):
    """dataset factory

    Args:
        args (argparse): args

    Returns:
        DataLoader: training set loader
        DataLoader: validation set loader
    """

    transform = transform_factory(args)

    if args.dataset_name == "CIFAR10":
        train_dataset = CIFAR10(
            root=args.root,
            train=True,
            download=True,
            transform=transform)
        val_set = CIFAR10(
            root=args.root,
            train=False,
            download=True,
            transform=transform)
        n_classes = 10

    elif args.dataset_name == "ImageFolder":
        root_train = os.path.join(args.root, args.train_dir)
        root_val = os.path.join(args.root, args.val_dir)
        assert os.path.exists(root_train)
        assert os.path.exists(root_val)
        assert os.path.isdir(root_train)
        assert os.path.isdir(root_val)

        train_dataset = ImageFolder(
            root=root_train,
            transform=transform)
        val_dataset = ImageFolder(
            root=root_val,
            transform=transform)
        assert len(train_dataset.classes) == len(val_dataset.classes)
        n_classes = len(train_dataset.classes)

    else:
        raise ValueError("invalid args.dataset_name")

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
