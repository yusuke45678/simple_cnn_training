import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet18,
    ResNet18_Weights,
)
import pytorch_lightning as pl

from utils import accuracy


class MyLightningModel(pl.LightningModule):
    def __init__(self, args, n_classes):
        super().__init__()
        self.args = args

        if args.use_pretrained:
            # Specity the directory where a pre-trained model is stored.
            # Otherwise, by default, models are stored in users home dir `~/.torch`
            os.environ['TORCH_HOME'] = args.torch_home

        if args.model == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if args.use_pretrained else None
            self.model = resnet18(weights=weights)
            self.model.fc = nn.Linear(
                self.model.fc.in_features, n_classes)
        elif args.model == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if args.use_pretrained else None
            self.model = resnet50(weights=weights)
            self.model.fc = nn.Linear(
                self.model.fc.in_features, n_classes)
        else:
            raise ValueError("invalid args.model")

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr, betas=self.args.betas)
        else:
            raise ValueError("invalid args.optimizer")

        if self.args.use_scheduler:
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=7, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        """training loop for one batch (not for one epoch)

        ``.to(device)`` and ``.train()`` are NOT needed in the code.

        Args:
            batch (tuple): batch as tuple(data, label)
            batch_idx (int): index of the batch in the epoch

        Returns:
            float: loss
        """

        data, labels = batch  # (BCHW, B)
        batch_size = data.size(0)

        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        self.log_dict(
            {
                'train_loss': loss,
                'train_top1': top1,
                'train_top5': top5,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=False,
            batch_size=batch_size)

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

        data, labels = batch  # (BCHW, B)
        batch_size = data.size(0)

        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        self.log_dict(
            {
                'val_loss': loss,
                'val_top1': top1,
                'val_top5': top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size)

        return {
            'batch_prediction': outputs,
            'batch_label': labels
        }

    def validation_epoch_end(self, val_step_outputs):
        '''
        aggregating validation predicttions
        NOTE: NOT working for DDP! only for DP or single GPU

        Args:
            val_step_outputs (stack of dict):
                a stack of all outputs of validation_step()
        '''

        all_preds = torch.cat([
            out['batch_prediction'] for out in val_step_outputs
        ])
        all_labels = torch.cat([
            out['batch_label'] for out in val_step_outputs
        ])

        self.loggers[0].experiment.log_confusion_matrix(
            all_labels.cpu().numpy(),
            all_preds.cpu().numpy(),
            step=self.global_step,
            epoch=self.current_epoch,
            title='Confusion matrix',
            row_label='Actual label',
            column_label='Prediction',
            # labels=[ ... list of category names ... ]
        )
