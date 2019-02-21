# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/19 16:06
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch.optim as optim

from tqdm import tqdm
from network.network import PlainNet, DeformNet, DeformNet_v2

from utils import utils


def parse_command():
    model_names = ['plain', 'deform', 'deform_v2']

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH')
    parser.add_argument('--model', type=str, default='plain', choices=model_names)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu', default=None, type=str, help='if not none, use Single GPU')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    return args


def create_mnist_loader(args):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def init_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def main():
    args = parse_command()
    print(args)

    # if setting gpu id, the using single GPU
    if args.gpu:
        print('Single GPU Mode.')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    train_loader, test_loader = create_mnist_loader(args)

    # create save dir and logger
    save_dir = utils.get_save_path(args)
    utils.write_config_file(args, save_dir)
    logger = utils.get_logger(save_dir)

    best_result = 0.0
    best_txt = os.path.join(save_dir, 'best.txt')

    train_acc = 0.0
    train_loss = 0.0

    start_epoch = 0
    start_iter = len(train_loader) * start_epoch + 1
    max_iter = len(train_loader) * (args.epochs - start_epoch + 1) + 1
    iter_save = len(train_loader)

    if args.model == 'plain':
        model = PlainNet()
    elif args.model == 'deform':
        model = DeformNet()
    else:
        model = DeformNet_v2()
    model.apply(init_weight)
    # You can use DataParallel() whether you use Multi-GPUs or not
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.NLLLoss()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=args.lr_patience)

    model.train()

    for it in tqdm(range(start_iter, max_iter + 1), total=max_iter, leave=False, dynamic_ncols=True):
        optimizer.zero_grad()

        try:
            input, target = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            input, target = next(loader_iter)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        per_acc = pred.eq(target.data.view_as(pred)).sum()
        train_acc += per_acc.cpu()

        if it % args.print_freq == 0:
            print('=> output: {}'.format(save_dir))
            print('Train Iter: [{0}/{1}]\t'
                  'Loss={Loss:.5f} '
                  'Accuracy={Acc:.5f}'
                  .format(it, max_iter, Loss=loss, Acc=float(per_acc) / args.batch_size))
            logger.add_scalar('Train/Loss', loss, it)
            # logger.add_scalar('Train/Acc', per_acc / args.batch_size, it)

        if it % iter_save == 0:
            epoch = it // iter_save
            correct, test_loss = test(model, test_loader, it, logger)

            # save the change of learning_rate
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                logger.add_scalar('Lr/lr_' + str(i), old_lr, it)

            # remember change of train/test loss and train/test acc
            train_loss = float(train_loss)
            train_acc = float(train_acc)
            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)

            logger.add_scalars('TrainVal/acc', {'train_acc': train_acc, 'test_acc': correct}, epoch)
            logger.add_scalars('TrainVal/loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)

            train_loss = 0.0
            train_acc = 0.0

            # remember best rmse and save checkpoint
            is_best = correct > best_result
            if is_best:
                best_result = correct
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}, acc={}".format(epoch, correct))

            scheduler.step(correct)

            model.train()

    logger.close()


def test(model, test_loader, epoch, logger=None):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum().cpu()

    test_loss = float(test_loss)
    correct = float(correct)
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(test_loss, 100. * correct))

    logger.add_scalar('Test/loss', test_loss, epoch)
    logger.add_scalar('Test/acc', correct, epoch)

    return float(correct), float(test_loss)


if __name__ == '__main__':
    main()
