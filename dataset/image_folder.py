import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def image_folder(args, train_transform, val_transform):

    root_train = os.path.join(args.root, args.train_dir)
    root_val = os.path.join(args.root, args.val_dir)
    assert os.path.exists(root_train) and os.path.isdir(root_train)
    assert os.path.exists(root_val) and os.path.isdir(root_val)

    train_dataset = ImageFolder(
        root=root_train,
        transform=train_transform)
    val_dataset = ImageFolder(
        root=root_val,
        transform=val_transform)

    assert sorted(train_dataset.classes) == sorted(val_dataset.classes)
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

    return train_loader, val_loader, n_classes
