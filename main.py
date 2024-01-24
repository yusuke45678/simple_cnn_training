import torch
from tqdm import tqdm

from args import ArgParse
from dataset import configure_dataloader, DataloaderConfig
from model import configure_model, ModelConfig
from setup import (
    configure_optimizer, OptimizerConfig,
    configure_scheduler, SchedulerConfig
)
from logger import configure_logger, LoggerConfig
from utils import (
    save_to_checkpoint, save_to_comet,
    load_from_checkpoint,
)

from train import train, TrainConfig
from val import validation


def main():

    args = ArgParse.get()

    logger = configure_logger(LoggerConfig(
        logged_params=vars(args),
        model_name=args.model_name,
        disable_logging=args.disable_comet
    ))

    dataloaders = configure_dataloader(DataloaderConfig(
        command_line_args=args,
        dataset_name=args.dataset_name
    ))

    model = configure_model(ModelConfig(
        model_name=args.model_name,
        use_pretrained=args.use_pretrained,
        torch_home=args.torch_home,
        n_classes=dataloaders.n_classes,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        gpu_strategy=args.gpu_strategy
    ))

    optimizer = configure_optimizer(OptimizerConfig(
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        betas=args.betas,
        model_params=model.get_parameters()
    ))
    scheduler = configure_scheduler(SchedulerConfig(
        optimizer=optimizer,
        use_scheduler=args.use_scheduler
    ))

    train_config = TrainConfig(
        grad_accum_interval=args.grad_accum,
        log_interval_steps=args.log_interval_steps
    )

    if args.checkpoint_to_resume:
        (
            start_epoch,
            current_train_step,
            current_val_step,
            loaded_model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(
            args.checkpoint_to_resume,
            model.get_model(),
            optimizer,
            scheduler,
            model.get_device()
        )
        model.set_model(loaded_model)
    else:
        current_train_step = 1
        current_val_step = 1
        start_epoch = 0

    with tqdm(range(start_epoch + 1, args.num_epochs + 1)) as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[Epoch {current_epoch}]")

            train_output = train(
                model,
                optimizer,
                scheduler,
                dataloaders.train_loader,
                current_train_step,
                current_epoch,
                logger,
                train_config
            )
            current_train_step = train_output.train_step

            if (
                current_epoch % args.val_interval_epochs == 0
                or current_epoch == args.num_epochs
            ):
                val_output = validation(
                    model,
                    dataloaders.val_loader,
                    current_val_step,
                    current_epoch,
                    logger,
                )
                current_val_step = val_output.val_step

                checkpoint_dict = save_to_checkpoint(
                    args.save_checkpoint_dir,
                    current_epoch,
                    current_train_step,
                    current_val_step,
                    val_output.top1,
                    model.get_model(),
                    optimizer,
                    scheduler,
                    logger
                )
                save_to_comet(
                    checkpoint_dict,
                    args.model_name,
                    logger
                )

            if scheduler:
                scheduler.update()


if __name__ == "__main__":
    main()
