from dataclasses import dataclass
import argparse

from dataset import (
    cifar10,
    Cifar10Info,
    image_folder,
    ImageFolderInfo,
    video_folder,
    transform_image,
    transform_video
)


@dataclass
class DatasetInfo:
    command_line_args: argparse.Namespace
    dataset_name: str


def dataloader_factory(
    dataset_info: DatasetInfo
):
    """dataset factory

    Args:
        dataset_info (DatasetInfo): information for dataset factory

    Raises:
        ValueError: invalide dataset name given by command line

    Returns:
        torch.utils.data.DataLoader: training set loader
        torch.utils.data.DataLoader: validation set loader
        int: number of classes
    """
    args = dataset_info.command_line_args

    if dataset_info.dataset_name == "CIFAR10":
        train_transform, val_transform = transform_image()
        train_loader, val_loader, n_classes = \
            cifar10(Cifar10Info(
                root=args.root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_info.dataset_name == "ImageFolder":
        train_transform, val_transform = transform_image()
        train_loader, val_loader, n_classes = \
            image_folder(ImageFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_info.dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_video(dataset_info.command_line_args)
        train_loader, val_loader, n_classes = \
            video_folder(
                dataset_info.command_line_args,
                train_transform,
                val_transform
            )

    else:
        raise ValueError("invalid dataset_name")


    return train_loader, val_loader, n_classes
