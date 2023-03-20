import pytorch_lightning as pl

from transforms import transform_image, transform_video
from dataset_factory import cifar10, image_folder, video_folder


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.dataset_name == 'CIFAR10':
            train_transform, val_transform = transform_image(args)
            self.train_loader, self.val_loader, self.n_classes = \
                cifar10(args, train_transform, val_transform)

        elif args.dataset_name == 'ImageFolder':
            train_transform, val_transform = transform_image(args)
            self.train_loader, self.val_loader, self.n_classes = \
                image_folder(args, train_transform, val_transform)

        elif args.dataset_name == 'VideoFolder':
            train_transform, val_transform = transform_video(args)
            self.train_loader, self.val_loader, self.n_classes = \
                video_folder(args, train_transform, val_transform)

        else:
            raise ValueError('invalid args.dataset_name')

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
