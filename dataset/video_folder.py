import os
import itertools
from pathlib import Path
from typing import Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    RandomSampler,
)

from torchvision.transforms import v2 as transforms

from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.data.clip_sampling import (
    RandomClipSampler,
    ConstantClipsPerVideoSampler,
)

from .my_videofolder import MyVideoDataset


@dataclass
class VideoFolderInfo:
    root: str
    train_dir: str
    val_dir: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms
    clip_duration: float
    clips_per_video: int


def collate_for_video(batch: Any) -> Tuple[Any, Any]:
    batch_dict = torch.utils.data.default_collate(batch)
    return batch_dict['video'], batch_dict['label']


def video_folder(
        video_folder_info: VideoFolderInfo
) -> Tuple[DataLoader, DataLoader, int]:
    """creating dataloaders for videos in folders by pytorchvideo

    Args:
        video_folder_info (VideoFolderInfo): information for dataloaders

    Returns:
        DataLoader: train dataloader
        DataLoader: val dataloader
        int: number of classes
    """

    root_train_dir = os.path.join(video_folder_info.root, video_folder_info.train_dir)
    root_val_dir = os.path.join(video_folder_info.root, video_folder_info.val_dir)

    assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
    assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

    # train_dataset = labeled_video_dataset(
    #     data_path=root_train_dir,
    #     clip_sampler=ConstantClipsPerVideoSampler(
    #         clip_duration=video_folder_info.clip_duration,
    #         clips_per_video=video_folder_info.clips_per_video
    #     ),
    #     transform=video_folder_info.train_transform,
    #     decode_audio=False,
    #     decoder="pyav",
    # )
    # val_dataset = labeled_video_dataset(
    #     data_path=root_val_dir,
    #     clip_sampler=ConstantClipsPerVideoSampler(
    #         clip_duration=video_folder_info.clip_duration,
    #         clips_per_video=video_folder_info.clips_per_video
    #     ),
    #     video_sampler=SequentialSampler,
    #     transform=video_folder_info.val_transform,
    #     decode_audio=False,
    #     decoder="pyav",
    # )
    # 自作ローダーの追加
    train_dataset = MyVideoDataset(
        root_dir=root_train_dir,
        num_frames=16,
        transform=video_folder_info.train_transform)
    val_dataset = MyVideoDataset(
        root_dir=root_val_dir,
        num_frames=16,
        transform=video_folder_info.val_transform)

    train_dataset.classes = sorted([d.name for d in Path(root_train_dir).iterdir()])
    val_dataset.classes = sorted([d.name for d in Path(root_val_dir).iterdir()])

    # テスト
    # train_dataset.classes.sort()
    # val_dataset.classes.sort()
    print(train_dataset.classes[0])
    print(val_dataset.classes[0])

    # test = train_dataset.classes.pop(0)
    assert train_dataset.classes == val_dataset.classes

    train_dataset.n_classes = len(train_dataset.classes)
    val_dataset.n_classes = len(val_dataset.classes)
    assert train_dataset.n_classes == val_dataset.n_classes

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=video_folder_info.batch_size,
    #     num_workers=video_folder_info.num_workers,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=video_folder_info.batch_size,
    #     num_workers=video_folder_info.num_workers,
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=video_folder_info.batch_size,
        drop_last=True,
        num_workers=video_folder_info.num_workers,
        # num_workers=video_folder_info.num_workers,
        collate_fn=collate_for_video,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=video_folder_info.batch_size,
        drop_last=False,
        num_workers=video_folder_info.num_workers,
        collate_fn=collate_for_video,
        shuffle=True
    )
#
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
        # return len(self.video_paths)
