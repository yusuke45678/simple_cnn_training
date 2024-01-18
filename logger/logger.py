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
        comet_ml.Experiment: Experiment object
    """

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    experiment = Experiment(disabled=logger_info.disable_logging)

    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    experiment.set_name(exp_name)
    experiment.add_tag(logger_info.model_name)

    experiment.log_parameters(logger_info.logged_params)

    return experiment
