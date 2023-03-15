from datetime import datetime
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def logger_factory(args):

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')

    # logger[0]
    comet_logger = CometLogger(
        # Use ./.comet.config and ~/.comet.config
        # to specify API key, workspace and project name.
        # DO NOT put API key in the code!
        save_dir=args.comet_log_dir,
        experiment_name=exp_name,
        parse_args=True,
    )
    comet_logger.experiment.add_tag(args.model)

    # logger[1]
    tb_logger = TensorBoardLogger(
        save_dir=args.tf_log_dir,
        name=exp_name
    )

    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_checkpoint_dir,
        monitor='val_top1',
        mode='max',  # larger acc is better
        save_top_k=2,
        filename='epoch{epoch}_steps{step}_acc={val_top1:.2f}',
    )

    return (comet_logger, tb_logger), checkpoint_callback
