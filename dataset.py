from transforms import transform_image, transform_video
from dataset_factory import cifar10, image_folder, video_folder


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
        transform_func = transform_image
        dataset_func = cifar10

    elif args.dataset_name == "ImageFolder":
        transform_func = transform_image
        dataset_func = image_folder

    elif args.dataset_name == "VideoFolder":
        transform_func = transform_video
        dataset_func = video_folder

    else:
        raise ValueError("invalid args.dataset_name")

    train_transform, val_transform = transform_func(args)
    train_loader, val_loader, n_classes = dataset_func(
        args, train_transform, val_transform
    )

    return train_loader, val_loader, n_classes
