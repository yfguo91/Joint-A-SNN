import torch
import os
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data.autoaugment import CIFAR10Policy, Cutout
from data.sampler import DistributedSampler
import time
from models import *
from torch.utils.data import DataLoader
from optim.dist_helper import DistModule, dist_init, dist_finalize, allaverage, save_file
from optim.log_helper import create_logger, get_logger
import linklink as link


def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='/mnt/lustre/share/prototype_cifar/cifar10/',
                                train=True, download=False, transform=transform_train)
        val_dataset = CIFAR10(root='/mnt/lustre/share/prototype_cifar/cifar10/',
                              train=False, download=False, transform=transform_test)

    else:
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
        train_dataset = CIFAR100(root='/mnt/lustre/share/prototype_cifar/cifar100/',
                                 train=True, download=False, transform=transform_train)
        val_dataset = CIFAR100(root='/mnt/lustre/share/prototype_cifar/cifar100/',
                               train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, round_up=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


if __name__ == '__main__':

    dist_init()

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res20m', type=str, help='dataset name',
                        choices=['CIFARNet', 'VGG16', 'res19', 'res20', 'res20m'])
    parser.add_argument('--batch_size', default=10, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=1, type=int, help='snn threshold')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--shift_snn', default=100, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=10, type=int, help='snn step')
    parser.add_argument('--spike', action='store_true', help='use spiking network')
    parser.add_argument('--teacher', action='store_true', help='use teacher to do distillation')

    args = parser.parse_args()

    create_logger(os.path.join('', 'log.txt'))
    logger = get_logger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    use_cifar10 = args.dataset == 'CIFAR10'

    train_loader, test_loader = build_data(batch_size=args.batch_size)
    best_acc = 0
    best_epoch = 0

    name = 'snn_T{}'.format(args.step) if args.spike is True else 'ann'
    model_save_name = 'raw/' + name + '_' + args.arch + '_wd1e4_cifardvs.pth'

    snn = resnet18(num_classes=10 if use_cifar10 else 100, input_c=3)
    teacher = None

    snn.cuda()
    device = next(snn.parameters()).device
    ann = DistModule(snn, sync=True)

    num_epochs = 300
    criterion = nn.CrossEntropyLoss().to(device)
    # build optimizer
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    temp_list = []
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.view((-1)).to(device)
            images = images.to(device)

            loss = ann.module.train_forward(images, labels)
            running_loss += loss.item()
            loss.backward()
            ann.sync_gradients()
            optimizer.step()
            if (i + 1) % 80 == 0:
                logger.info('Time elapsed: {}'.format(time.time() - start_time))
        scheduler.step()

        correct = torch.Tensor([0.]).cuda()
        total = torch.Tensor([0.]).cuda()
        acc = torch.Tensor([0.]).cuda()

        # start testing
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.view((-1)).to(device)
                outputs = ann(inputs)
                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        acc = allaverage(acc)
        logger.info('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
        if best_acc < acc and epoch > 10:
            best_acc = acc
            best_epoch = epoch + 1
            # save_file(ann.state_dict(), model_save_name)
        logger.info('best_acc is: {}'.format(best_acc))
        logger.info('Iters: {}\n\n'.format(epoch))

    logger.info('{}'.format(temp_list))
    dist_finalize()
