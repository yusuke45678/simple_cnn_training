import torch
import torch.nn as nn
from tqdm import tqdm

from args import get_args
from dataset import dataset_facory
from model import model_factory
from optimizer import optimizer_factory, scheduler_factory
from logger import logger_factory
from train import train, val
from utils import save_to_checkpoint, load_from_checkpoint


def main():
    args = get_args()

    experiment = logger_factory(args)

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(args, model)
    scheduler = scheduler_factory(args, optimizer)

    global_step = 1
    start_epoch = 0

    if args.resume_from_checkpoint:
        (
            start_epoch,
            global_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(args, model, optimizer, scheduler, device)

    if args.gpu_strategy == "dp":
        model = nn.DataParallel(model)

    with tqdm(range(start_epoch + 1, args.num_epochs + 1)) as pbar_epoch:
        for current_epoch in pbar_epoch:
            pbar_epoch.set_description(f"[Epoch {current_epoch}]")

            global_step = train(
                model,
                criterion,
                optimizer,
                train_loader,
                device,
                global_step,
                current_epoch,
                experiment,
                args,
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
                    args,
                )

                save_to_checkpoint(
                    args,
                    current_epoch,
                    global_step,
                    val_top1,
                    model,
                    optimizer,
                    scheduler,
                    experiment,
                )

            if args.use_scheduler:
                scheduler.update()


if __name__ == "__main__":
    main()
