from dataclasses import dataclass
import argparse

from torch.utils.data import DataLoader

from dataset import (
    cifar10,
    Cifar10Info,
    image_folder,
    ImageFolderInfo,
    video_folder,
    VideoFolderInfo,
    zero_images,
    ZeroImageInfo,
    transform_image,
    TransformImageInfo,
    transform_video,
    TransformVideoInfo,
)


@dataclass
class DataloaderConfig:
    command_line_args: argparse.Namespace
    dataset_name: str


@dataclass
class DataloadersInfo:
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int


def configure_dataloader(
    dataloader_info: DataloaderConfig
):
    """dataloader factory

    Args:
        dataloader_info (DataloaderInfo): information for dataset factory

    Raises:
        ValueError: invalide dataset name given by command line

    Returns:
        torch.utils.data.DataLoader: training set loader
        torch.utils.data.DataLoader: validation set loader
        int: number of classes
    """
    args = dataloader_info.command_line_args

    if dataloader_info.dataset_name == "CIFAR10":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            cifar10(Cifar10Info(
                root=args.root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataloader_info.dataset_name == "ImageFolder":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
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

    elif dataloader_info.dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_video(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            video_folder(VideoFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform,
                clip_duration=args.clip_duration,
                clips_per_video=args.clips_per_video
            ))

    elif dataloader_info.dataset_name == "ZeroImages":
        train_transform, _ = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            zero_images(ZeroImageInfo(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                transform=train_transform,
            ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )
