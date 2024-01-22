from dataclasses import dataclass
from typing import Tuple

import numpy as np

from torch.utils.data import (
    DataLoader,
    Dataset
)
from torchvision import transforms


@dataclass
class ImageZeroDummyInfo():
    batch_size: int
    num_workers: int
    transform: transforms
    image_size: int = 265
    dataset_size: int = 1000
    n_classes: int = 10


class ImageZeroDummyDataset(Dataset):
    def __init__(self, dataset_size, image_size, n_classes, transform=None):
        self.transform = transform
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.n_classes = n_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image = np.zeros((3, self.image_size, self.image_size))
        label = index % self.n_classes
        if self.transform:
            image = self.transform(image)
        return image, label


def image_zero_dummy(
        image_zero_dummy_info: ImageZeroDummyInfo
) -> Tuple[DataLoader, DataLoader, int]:

    train_loader = DataLoader(
        ImageZeroDummyDataset(
            dataset_size=image_zero_dummy_info.dataset_size,
            image_size=image_zero_dummy_info.image_size,
            transform=image_zero_dummy_info.transform,
            n_classes=image_zero_dummy_info.n_classes
        ),
        batch_size=image_zero_dummy_info.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=image_zero_dummy_info.num_workers
    )
    val_loader = DataLoader(
        ImageZeroDummyDataset(
            dataset_size=image_zero_dummy_info.dataset_size,
            image_size=image_zero_dummy_info.image_size,
            transform=image_zero_dummy_info.transform,
            n_classes=image_zero_dummy_info.n_classes
        ),
        batch_size=image_zero_dummy_info.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=image_zero_dummy_info.num_workers
    )

    return train_loader, val_loader, image_zero_dummy_info.n_classes
