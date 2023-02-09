import argparse
import shutil
import os
import time
import torch
import logging as logger
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from models.resnet import resnet18
from functions import seed_all
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100


parser = argparse.ArgumentParser(description='PyTorch Joint Training of ANN and SNN')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T', '--time', default=4, type=int, metavar='N',
                    help='snn simulation time (default: 4)')
parser.add_argument('--amp', action='store_true',
                    help='if use amp training.')
args = parser.parse_args()


def build_cifar(use_cifar10=True, download=True, normalize=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    aug.append(transforms.ToTensor())

    if use_cifar10:
        if normalize:
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )

        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='data',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='data',
                              train=False, download=download, transform=transform_test)

    else:
        if normalize:

            aug.append(
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='data',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='data',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset


def train(model, device, train_loader, optimizer, epoch, scaler, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    s_time = time.time() 
    model.forward = model.train_forward
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            with torch.autograd.set_detect_anomaly(True):
                loss = model(images, labels)
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        total += float(labels.size(0))

    e_time = time.time()
    return running_loss / total, 100 * correct / total, (e_time-s_time)/60


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    model.forward = model.test_forward
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':

    seed_all(args.seed)

    train_dataset, val_dataset = build_cifar(use_cifar10=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    model = resnet18(num_classes=10)

    model.T = args.time
    model.cuda()
    device = next(model.parameters()).device

    scaler = GradScaler() if args.amp else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/256 * args.batch_size, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    best_acc = 0
    best_epoch = 0
    print('start training!')
    for epoch in range(args.epochs):

        loss, acc, t_diff = train(model, device, train_loader, optimizer, epoch, scaler, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f},\t time elapsed: {}'.format(epoch, args.epochs, loss, acc,
                                                                                    t_diff))
        scheduler.step()
        facc = test(model, test_loader, device)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
        print('Best Test acc={:.3f}'.format(best_acc))