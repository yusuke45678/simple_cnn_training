import torch
import lightning.pytorch as pl

from args import get_args
from dataset_pl import MyDataModule
from model_pl import MyLightningModel
from logger_pl import logger_factory
from callback_pl import callback_factory

from lightning.pytorch.plugins import TorchSyncBatchNorm


def main():
    args = get_args()
    loggers, exp_name = logger_factory(args)
    callbacks = callback_factory(args)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    data_module = MyDataModule(args)
    model_lightning = MyLightningModel(args, data_module.n_classes, exp_name)

    strategy = "auto"
    # if args.gpu_strategy == 'dp':
    #     strategy = 'dp'
    if args.gpu_strategy == "ddp":
        # strategy = 'ddp_find_unused_parameters_false'
        strategy = "ddp"

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=0,
        # precision=16,  # for FP16 training
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5, # only for debug
        # limit_train_batches=5, # only for debug
        # limit_val_batches=5, # only for debug
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
        # profiler="simple",
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
