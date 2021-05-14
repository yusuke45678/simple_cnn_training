
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import dataset_facory
from model import model_factory
from args import get_args
from optimizer import optimizer_factory, scheduler_factory


def val(model, criterion, optimizer, loader, device):

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


def train(model, criterion, optimizer, loader, device):

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


def main():

    args = get_args()

    train_loader, val_loader, n_classes = dataset_facory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args, n_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(args, model)
    scheduler = scheduler_factory(args, optimizer)

    with tqdm(range(args.num_epochs)) as pbar_epoch:
        for e in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(e))

            train(model, criterion, optimizer, train_loader, device)

            if e % args.val_epochs:
                val(model, criterion, optimizer, val_loader, device)

            if args.use_scheduler:
                scheduler.update()


if __name__ == "__main__":
    main()
