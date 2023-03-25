from datetime import datetime
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger


def logger_factory(args):
    """generating two loggers.
    - comet logger for cloud logging
    - tensorboard logger for local logging (in case of network lost)

    Args:
        args (argparse): args

    Returns:
        Tuple[pytorch_lightning.loggers]: loggers
    """

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')

    # logger[0]
    comet_logger = CometLogger(
        # Use ./.comet.config and ~/.comet.config
        # to specify API key, workspace and project name.
        # DO NOT put API key in the code!
        save_dir=args.comet_log_dir,
        experiment_name=exp_name,
        parse_args=True,
        disabled=args.disable_comet,
    )
    comet_logger.experiment.add_tag(args.model)

    # logger[1]
    tb_logger = TensorBoardLogger(
        save_dir=args.tf_log_dir,
        name=exp_name
    )

    return comet_logger, tb_logger
