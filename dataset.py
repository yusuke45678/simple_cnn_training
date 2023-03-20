from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
import os

from torchvision import transforms
from torchvision.transforms import Compose

import torchvision.transforms as it  # image trans
import pytorchvideo.transforms as vt  # video trans

from pytorchvideo.data import labeled_video_dataset

from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from pytorchvideo.data.clip_sampling import (
    RandomClipSampler,
    ConstantClipsPerVideoSampler,
)


def transform_video(args):
    """transform for video clips

    Args:
        args (argparse): args

    Returns:
        Tuple[pytorchvideo.transforms]: train and val transforms
    """

    train_transform = Compose([
        vt.ApplyTransformToKey(
            key="video",
            transform=Compose([
                vt.UniformTemporalSubsample(args.frames_per_clip),
                it.Lambda(lambda x: x / 255.),
                vt.Normalize(
                    [0.45, 0.45, 0.45],
                    [0.225, 0.225, 0.225]
                ),
                vt.RandomShortSideScale(min_size=256, max_size=320,),
                it.RandomCrop(224),
                it.RandomHorizontalFlip(),
            ]),
        ),
        vt.RemoveKey("audio"),
    ])

    val_transform = Compose([
        vt.ApplyTransformToKey(
            key="video",
            transform=Compose([
                vt.UniformTemporalSubsample(args.frames_per_clip),
                it.Lambda(lambda x: x / 255.),
                vt.Normalize(
                    [0.45, 0.45, 0.45],
                    [0.225, 0.225, 0.225]
                ),
                vt.ShortSideScale(256),
                it.CenterCrop(224),
            ]),
        ),
        vt.RemoveKey("audio"),
    ])

    return train_transform, val_transform


def transform_image(args):
    """transform for images

    Args:
        args (argparse): args

    Returns:
        Tuple[torchvision.transforms]: train and val transforms
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def dataset_facory(args):
    """dataset factory

    Args:
        args (argparse): args

    Raises:
        ValueError: invalide dataset name given by command line

    Returns:
        torch.utils.data.DataLoader: training set loader
        torch.utils.data.DataLoader: validation set loader
        int: number of classes
    """

    if args.dataset_name == "CIFAR10":
        train_transform, val_transform = transform_image(args)

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


    elif args.dataset_name == "ImageFolder":
        train_transform, val_transform = transform_image(args)

        root_train = os.path.join(args.root, args.train_dir)
        root_val = os.path.join(args.root, args.val_dir)
        assert os.path.exists(root_train)
        assert os.path.exists(root_val)
        assert os.path.isdir(root_train)
        assert os.path.isdir(root_val)

        train_dataset = ImageFolder(
            root=root_train,
            transform=train_transform)
        val_dataset = ImageFolder(
            root=root_val,
            transform=val_transform)
        assert len(train_dataset.classes) == len(val_dataset.classes)
        n_classes = len(train_dataset.classes)

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

    elif args.dataset_name == "VideoFolder":
        train_transform, val_transform = transform_video(args)

        root_train = os.path.join(args.root, args.train_dir)
        root_val = os.path.join(args.root, args.val_dir)
        assert os.path.exists(root_train)
        assert os.path.exists(root_val)
        assert os.path.isdir(root_train)
        assert os.path.isdir(root_val)

        train_dataset = labeled_video_dataset(
            data_path=root_train,
            clip_sampler=RandomClipSampler(
                clip_duration=args.clip_duration,
            ),
            video_sampler=RandomSampler,
            transform=train_transform,
            decode_audio=False,
            decoder="pyav",
        )
        val_dataset = labeled_video_dataset(
            data_path=root_val,
            clip_sampler=ConstantClipsPerVideoSampler(
                clip_duration=args.clip_duration,
                clips_per_video=args.clips_per_video
            ),
            video_sampler=SequentialSampler,
            transform=val_transform,
            decode_audio=False,
            decoder="pyav",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers)

    else:
        raise ValueError("invalid args.dataset_name")

    return train_loader, val_loader, n_classes
