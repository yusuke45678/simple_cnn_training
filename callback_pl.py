from lightning.pytorch.callbacks import ModelSummary


def callback_factory(args):
    """generating callbacks

    Args:
        args (argparse): args

    Returns:
        List[pl.callbacks]: callbacks
    """

    callbacks = [
        ModelSummary(max_depth=3)
    ]

    return callbacks
