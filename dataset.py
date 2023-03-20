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
        train_transform, val_transform = transform_image(args)
        train_loader, val_loader, n_classes = \
            cifar10(args, train_transform, val_transform)

    elif args.dataset_name == "ImageFolder":
        train_transform, val_transform = transform_image(args)
        train_loader, val_loader, n_classes = \
            image_folder(args, train_transform, val_transform)

    elif args.dataset_name == "VideoFolder":
        train_transform, val_transform = transform_video(args)
        train_loader, val_loader, n_classes = \
            video_folder(args, train_transform, val_transform)

    else:
        raise ValueError("invalid args.dataset_name")

    return train_loader, val_loader, n_classes
