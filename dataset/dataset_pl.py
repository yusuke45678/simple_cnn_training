import lightning.pytorch as pl
from dataset import configure_dataloader, DatasetInfo


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train_loader, self.val_loader, self.n_classes = \
            configure_dataloader(DatasetInfo(
                command_line_args=args,
                dataset_name=args.dataset_name
            ))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
