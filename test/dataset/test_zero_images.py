import pytest
import torch

from dataset import (
    zero_images,
    ZeroImageInfo,
    transform_image,
    TransformImageInfo,
)


@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('num_workers', [1, 2])
def test_zero_images(
    batch_size,
    num_workers,
):

    train_transform, _ = \
        transform_image(TransformImageInfo())
    train_loader, val_loader, n_classes = \
        zero_images(ZeroImageInfo(
            batch_size=batch_size,
            num_workers=num_workers,
            transform=train_transform,
        ))

    assert n_classes > 1

    for loader in [train_loader, val_loader]:
        for data, labels in loader:

            assert isinstance(data, torch.Tensor)
            assert data.ndim == 4  # BCHW
            assert data.shape[0] == batch_size
            assert data.shape[1] == 3  # 3 channels (RGB)
            assert data.shape[2] == data.shape[3]  # H == W

            assert isinstance(labels, torch.Tensor)
            assert labels.ndim == 1
            assert labels.shape[0] == batch_size

            break  # check only one sample
