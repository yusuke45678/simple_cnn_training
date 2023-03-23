import os
from pathlib import Path
import itertools

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data import labeled_video_dataset
from torch.utils.data import (
    SequentialSampler,
    RandomSampler,
)
from pytorchvideo.data.clip_sampling import (
    RandomClipSampler,
    ConstantClipsPerVideoSampler,
)


def video_folder(args, train_transform, val_transform):

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

    train_dataset.classes = sorted([d.name for d in Path(root_train).iterdir()])
    val_dataset.classes = sorted([d.name for d in Path(root_val).iterdir()])
    assert train_dataset.classes == val_dataset.classes

    train_dataset.n_classes = len(train_dataset.classes)
    val_dataset.n_classes = len(val_dataset.classes)
    assert train_dataset.n_classes == val_dataset.n_classes

    train_loader = DataLoader(
        LimitDataset(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        LimitDataset(val_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers)

    return train_loader, val_loader, train_dataset.n_classes


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.

    https://github.com/facebookresearch/pytorchvideo/blob/f7e7a88a9a04b70cb65a564acfc38538fe71ff7b/tutorials/video_classification_example/train.py#L341
    https://github.com/facebookresearch/pytorchvideo/issues/96
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
