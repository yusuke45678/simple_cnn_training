
from tqdm import tqdm

import torch
import torch.nn as nn

import mlflow

from dataset import dataset_facory
from model import model_factory
from args import get_args
from optimizer import optimizer_factory, scheduler_factory


def val(model, criterion, optimizer, loader, device, iters, epoch):

    model.eval()
    loss_list = []
    acc_list = []
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

            loss_list.append(batch_loss)
            acc_list.append(batch_acc)

    val_loss = sum(loss_list) / len(loss_list)
    val_acc = sum(acc_list) / len(acc_list)

    mlflow.log_metrics({'val_loss': val_loss,
                        'val_acc': val_acc},
                       step=iters)


def train(model, criterion, optimizer, loader, device, iters, epoch):

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

            mlflow.log_metrics({'train_loss': batch_loss,
                                'train_acc': batch_acc},
                               step=iters)

            iters += 1

    return iters


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

    iters = 0

    with tqdm(range(args.num_epochs)) as pbar_epoch:

        for epoch in pbar_epoch:
            pbar_epoch.set_description('[Epoch {}]'.format(epoch))

            iters = train(model, criterion, optimizer, train_loader, device,
                          iters, epoch)

            if epoch % args.val_epochs:
                val(model, criterion, optimizer, val_loader, device,
                    iters, epoch)

            if args.use_scheduler:
                scheduler.update()


if __name__ == "__main__":
    main()
