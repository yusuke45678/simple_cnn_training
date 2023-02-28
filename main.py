import torch
import torch.nn as nn
from tqdm import tqdm

from args import get_args
from dataset import dataset_facory
from model import model_factory
from optimizer import optimizer_factory, scheduler_factory
from logger import logger_factory
from train import train, val


def main():

    args = get_args()

    experiment = logger_factory(args)

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_factory(args, n_classes)
    model = model.to(device)
    if args.use_dp:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(args, model)
    scheduler = scheduler_factory(args, optimizer)

    global_steps = 1

    with tqdm(range(1, args.num_epochs + 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(epoch))

            global_steps = train(
                model, criterion, optimizer, train_loader,
                device, global_steps, epoch, experiment, args)

            if (epoch + 1) % args.val_epochs == 0:
                val(model, criterion, val_loader, device,
                    global_steps, epoch, experiment)

            if args.use_scheduler:
                scheduler.update()


if __name__ == '__main__':
    main()
