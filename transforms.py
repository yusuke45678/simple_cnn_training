from torchvision import transforms
from torchvision.transforms import Compose

import torchvision.transforms as it  # image trans
import pytorchvideo.transforms as vt  # video trans


def transform_video(args):
    """transform for video clips

    Args:
        args (argparse): args

    Returns:
        Tuple[pytorchvideo.transforms]: train and val transforms
    """

    train_transform = Compose(
        [
            vt.ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        vt.UniformTemporalSubsample(args.frames_per_clip),
                        it.Lambda(lambda x: x / 255.0),
                        vt.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                        vt.RandomShortSideScale(
                            min_size=256,
                            max_size=320,
                        ),
                        it.RandomCrop(224),
                        it.RandomHorizontalFlip(),
                    ]
                ),
            ),
            vt.RemoveKey("audio"),
        ]
    )

    val_transform = Compose(
        [
            vt.ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        vt.UniformTemporalSubsample(args.frames_per_clip),
                        it.Lambda(lambda x: x / 255.0),
                        vt.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                        vt.ShortSideScale(256),
                        it.CenterCrop(224),
                    ]
                ),
            ),
            vt.RemoveKey("audio"),
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

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
