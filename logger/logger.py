from datetime import datetime
from comet_ml import Experiment
from dataclasses import dataclass


@dataclass
class LoggerInfo:
    logged_params: dict
    model_name: str
    disable_logging: bool


def logger_factory(
        logger_info: LoggerInfo
) -> Experiment:
    """comet logger factory

    Args:
        logger_info (LoggerInfo): information for logger

    Returns:
        comet_ml.Experiment: logger
    """

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    experiment_logger = Experiment(disabled=logger_info.disable_logging)

    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    experiment_logger.set_name(exp_name)
    experiment_logger.add_tag(logger_info.model_name)

    experiment_logger.log_parameters(logger_info.logged_params)

    return experiment_logger
