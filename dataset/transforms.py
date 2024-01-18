import torchvision.transforms as image_transform
import pytorchvideo.transforms as video_transform


def transform_video(args):
    """transform for video clips

    Args:
        args (argparse): args

    Returns:
        Tuple[pytorchvideo.transforms]: train and val transforms
    """

    train_transform = image_transform.Compose(
        [
            video_transform.ApplyTransformToKey(
                key="video",
                transform=image_transform.Compose(
                    [
                        video_transform.UniformTemporalSubsample(args.frames_per_clip),
                        image_transform.Lambda(lambda x: x / 255.0),
                        video_transform.Normalize(
                            [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
                        ),
                        video_transform.RandomShortSideScale(
                            min_size=256,
                            max_size=320,
                        ),
                        image_transform.RandomCrop(224),
                        image_transform.RandomHorizontalFlip(),
                    ]
                ),
            ),
            video_transform.RemoveKey("audio"),
        ]
    )

    val_transform = image_transform.Compose(
        [
            video_transform.ApplyTransformToKey(
                key="video",
                transform=image_transform.Compose(
                    [
                        video_transform.UniformTemporalSubsample(args.frames_per_clip),
                        image_transform.Lambda(lambda x: x / 255.0),
                        video_transform.Normalize(
                            [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
                        ),
                        video_transform.ShortSideScale(256),
                        image_transform.CenterCrop(224),
                    ]
                ),
            ),
            video_transform.RemoveKey("audio"),
        ]
    )

    return train_transform, val_transform


def transform_image(args):
    """transform for images

    Args:
        args (argparse): args

    Returns:
        Tuple[torchvision.transforms]: train and val transforms
    """

    train_transform = image_transform.Compose(
        [
            image_transform.RandomResizedCrop(224),
            image_transform.RandomHorizontalFlip(),
            image_transform.ToTensor(),
            image_transform.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = image_transform.Compose(
        [
            image_transform.Resize(256),
            image_transform.CenterCrop(224),
            image_transform.ToTensor(),
            image_transform.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    return train_transform, val_transform
