import os

import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from utils import accuracy
from model import model_factory
from optimizer import optimizer_factory, scheduler_factory


class MyLightningModel(pl.LightningModule):
    """Lightning model"""

    def __init__(self, args, n_classes, exp_names):
        """constructor

        Args:
            args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = args
        self.exp_name = exp_names

        self.model = model_factory(args, n_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        """optimizer and scheduler

        Returns:
            optim or dict: optimizer, or dict
        """

        optimizer = optimizer_factory(self.args, self.model)
        scheduler = scheduler_factory(self.args, optimizer)

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def configure_callbacks(self):
        """model-specific callbacks

        Returns:
            callback or List[callback]: callback(s)
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_checkpoint_dir,
            monitor="val_top1",
            mode="max",  # larger is better
            save_top_k=2,
            filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
            auto_insert_metric_name=False,
        )
        return checkpoint_callback

    def training_step(self, batch, batch_idx):
        """training loop for one batch (not for one epoch)

        ``.to(device)`` and ``.train()`` are NOT needed in the code.

        Args:
            batch (tuple): batch as tuple(data, label)
            batch_idx (int): index of the batch in the epoch

        Returns:
            float: loss
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        self.log_dict(
            {
                "train_loss": loss,
                "train_top1": top1,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """validation for one batch (not for one epoch)

        ``.to(device)``, ``torch.no_grad()``, and ``.eval()`` are
            NOT needed in the code.

        Args:
            batch (tuple): batch as tuple(data, label)
            batch_idx (int): index of the batch in the epoch

        Returns:
            dict: used by validation_epoch_end()
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        self.log_dict(
            {
                "val_loss": loss,
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )
