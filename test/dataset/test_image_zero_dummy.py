from dataset import (
    image_zero_dummy,
    ImageZeroDummyInfo,
    transform_image,
    TransformImageInfo,
)


def test_image_zero_dummy():
    batch_size = 8
    num_workers = 1

    train_transform, _ = \
        transform_image(TransformImageInfo())
    train_loader, val_loader, n_classes = \
        image_zero_dummy(ImageZeroDummyInfo(
            batch_size=batch_size,
            num_workers=num_workers,
            transform=train_transform,
        ))

    train_batch = next(iter(train_loader))
    assert train_batch.shape == batch_size

    val_batch = next(iter(val_loader))
    assert val_batch.shape == batch_size
