from datetime import datetime
from comet_ml import Experiment


def logger_factory(args):
    """comet logger factory

    Args:
        args (argparse): args

    Returns:
        comet_ml.Experiment: Experiment object
    """

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    experiment = Experiment(disabled=args.disable_comet)

    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    experiment.set_name(exp_name)
    experiment.add_tag(args.model)

    experiment.log_parameters(vars(args))

    return experiment
