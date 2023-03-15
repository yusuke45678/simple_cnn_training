import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

from args import get_args
from dataset_pl import MyDataModule
from model_pl import MyLightningModel
from logger_pl import get_loggers


def main():

    args = get_args()
    loggers, checkpoint_callback = get_loggers(args)

    assert torch.cuda.is_available(), 'cpu is not supported'

    data_module = MyDataModule(args)
    model_lightning = MyLightningModel(args)

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_false' if args.gpus > 1 else None,
        max_epochs=args.n_epochs,
        logger=loggers,
        log_every_n_steps=10,  # default:50
        # accumulate_grad_batches=args.grad_upd,
        # precision=16,
        num_sanity_val_steps=0,
        # #
        # # only for debug
        # #
        # fast_dev_run=True,
        # fast_dev_run=5,
        # limit_train_batches=5,
        # limit_val_batches=5,
        callbacks=[
            ModelSummary(max_depth=3),
            checkpoint_callback,
        ]
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=args.ckpt_path
    )


if __name__ == '__main__':
    main()
