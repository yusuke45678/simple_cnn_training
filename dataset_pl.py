import lightning.pytorch as pl

from dataset import dataset_facory


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train_loader, self.val_loader, self.n_classes = dataset_facory(args)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
