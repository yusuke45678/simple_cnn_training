
import argparse
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


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


def transform_factory(do_crop=True):
    if do_crop:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


def dataset_facory(args):
    """dataset factory

    Args:
        args (argparse): args

    Returns:
        train dataloader (DataLoader): training set loader
        val dataloader (DataLoader): validation set loader
    """

    transform = transform_factory()

    if args.dataset_name == "CIFAR10":
        train_set = CIFAR10(root=args.root,
                            train=True,
                            download=True,
                            transform=transform)
        val_set = CIFAR10(root=args.root,
                          train=False,
                          download=True,
                          transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    return train_loader, val_loader


def get_args():
    """generate argparse object

    Returns:
        args: [description]
    """
    parser = argparse.ArgumentParser(description='simple cnn model')
    parser.add_argument('-r', '--root', type=str, default='./data',
                        help='root of dataset. default to ./data')
    parser.add_argument('-m', '--models', type=str, default='./models',
                        help='TORCH_HOME where pre-trained models are stored.'
                        ' default to ./models')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10',
                        help='name of dataset. default to CIFAR10')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size. default to 8')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of workers. default to 2')
    parser.add_argument('-e', '--num_epochs', type=int, default=25,
                        help='number of epochs. default to 25')
    args = parser.parse_args()
    return args


def model_factory(args):
    os.environ['TORCH_HOME'] = args.models
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model


def main():

    args = get_args()

    train_loader, val_loader = dataset_facory(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args)
    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
