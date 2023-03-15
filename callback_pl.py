from pytorch_lightning.callbacks import ModelSummary


def callback_factory(args):

    callbacks = [
        ModelSummary(max_depth=3)
    ]

    return callbacks
