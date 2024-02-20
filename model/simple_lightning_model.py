import argparse
import os

from torch.nn import functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import compute_topk_accuracy
from model import configure_model, ModelConfig
from setup import configure_optimizer, configure_scheduler


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: argparse.Namespace,
            n_classes: int,
            exp_name: str,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name

        self.model = configure_model(ModelConfig(
            model_name=self.args.model_name,
            use_pretrained=self.args.use_pretrained,
            torch_home=self.args.torch_home,
            n_classes=n_classes,
        ))

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            model_params=self.model.parameters()
        )
        scheduler = configure_scheduler(
            optimizer=optimizer,
            use_scheduler=self.args.use_scheduler
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def forward(self, data, labels):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#forward-vs-training-step
        """
        return F.softmax(self.model(data, labels).logits)

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=False,  # do NOT sync log metrics for training
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=False,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        """

        # DO NOT USE .to() or model.train() here

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss
        # DO NOT USE loss.backward() here

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_train_loss_top15(loss, top1, top5, batch_size)

        return loss  # then loss.backward() will be done automatically by lightning

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_val_loss_top15(loss, top1, top5, batch_size)
