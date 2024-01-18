import torch
from torch import nn
from tqdm import tqdm

from args import get_args
from dataset import dataloader_factory, DatasetInfo
from model import model_factory, ModelInfo
from setup import (
    optimizer_factory,
    OptimizerInfo,
    scheduler_factory,
    SchedulerInfo
)
from logger import logger_factory, LoggerInfo
from train import train_one_epoch, TrainInfo
from val import val
from utils import (
    save_to_checkpoint,
    save_model_to_comet,
    load_from_checkpoint,
)


def main():

    args = get_args()

    experiment = logger_factory(LoggerInfo(
        logged_params=vars(args),
        model_name=args.model,
        disable_logging=args.disable_comet
    ))

    train_loader, val_loader, n_classes = \
        dataloader_factory(DatasetInfo(
            command_line_args=args,
            dataset_name=args.dataset_name
        ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(ModelInfo(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        torch_home=args.torch_home,
        n_classes=n_classes,
        device=device,
        gpu_strategy=args.gpu_strategy
    ))

    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer_factory(OptimizerInfo(
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        betas=args.betas,
        model_params=model.parameter()
    ))
    scheduler = scheduler_factory(SchedulerInfo(
        optimizer=optimizer,
        use_scheduler=args.use_scheduler
    ))

    global_step = 1
    start_epoch = 0
    train_info = TrainInfo(
        grad_accum_interval=args.grad_accum,
        log_interval_steps=args.log_interval_steps
    )

    if args.checkpoint_to_resume:
        (
            start_epoch,
            global_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(args.checkpoint_to_resume, model, optimizer, scheduler, device)

    with tqdm(range(start_epoch + 1, args.num_epochs + 1)) as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[Epoch {current_epoch}]")

            global_step = train_one_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                device,
                global_step,
                current_epoch,
                experiment,
                train_info
            )

            if (
                current_epoch % args.val_interval_epochs == 0
                or current_epoch == args.num_epochs
            ):
                _, val_top1 = val(
                    model,
                    criterion,
                    val_loader,
                    device,
                    global_step,
                    current_epoch,
                    experiment,
                )

                checkpoint_dict = save_to_checkpoint(
                    args.save_checkpoint_dir,
                    current_epoch,
                    global_step,
                    val_top1,
                    # model if not args.gpu_strategy == "dp" else model.module,
                    model,
                    optimizer,
                    scheduler,
                    experiment
                )
                save_model_to_comet(
                    checkpoint_dict,
                    args.model_name,
                    experiment
                )

            if scheduler:
                scheduler.update()


if __name__ == "__main__":
    main()
