
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import dataset_facory
from model import model_factory
from args import get_args


def val(model, criterion, optimizer, scheduler, loader, device):

    model.eval()

    with torch.no_grad(), tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[val]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            batch_loss = loss.item() / data.size(0)
            _, preds = output.max(1)
            batch_acc = preds.eq(labels).sum().item() / data.size(0)

            pbar_loss.set_postfix_str(
                'loss={:.05f}, acc={:.03f}'.format(batch_loss, batch_acc))


def train(model, criterion, optimizer, scheduler, loader, device):

    model.train()

    with tqdm(loader, leave=False) as pbar_loss:
        pbar_loss.set_description('[train]')
        for data, labels in pbar_loss:

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item() / data.size(0)
            _, preds = output.max(1)
            batch_acc = preds.eq(labels).sum().item() / data.size(0)

            pbar_loss.set_postfix_str(
                'loss={:.05f}, acc={:.03f}'.format(batch_loss, batch_acc))


def optimizer_factory(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.optmizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, betas=args.betas)
    else:
        raise ValueError("invalid args.optimizer")

    return optimizer


def main():

    args = get_args()

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args, n_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer_factory(args, model)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    with tqdm(range(args.num_epochs)) as pbar_epoch:
        for e in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(e))

            train(model, criterion, optimizer, scheduler,
                  train_loader, device)
            if e % 2:
                val(model, criterion, optimizer, scheduler,
                    val_loader, device)


if __name__ == "__main__":
    main()
