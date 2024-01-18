from dataclasses import dataclass
import argparse

from dataset import (
    cifar10,
    image_folder,
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

    if dataset_info.dataset_name == "CIFAR10":
        transform_func = transform_image
        dataset_func = cifar10

    elif dataset_info.dataset_name == "ImageFolder":
        transform_func = transform_image
        dataset_func = image_folder

    elif dataset_info.dataset_name == "VideoFolder":
        transform_func = transform_video
        dataset_func = video_folder

    else:
        raise ValueError("invalid dataset_name")

    train_transform, val_transform = \
        transform_func(dataset_info.command_line_args)

    train_loader, val_loader, n_classes = \
        dataset_func(
            dataset_info.command_line_args,
            train_transform,
            val_transform
        )

    return train_loader, val_loader, n_classes
