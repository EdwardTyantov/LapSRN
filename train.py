# -*- coding: utf-8 -*-
import sys, os, argparse, random
from math import log10
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#
from model import LasSRN
from data import get_training_set, get_test_set


def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3


def train(epoch, model, criterion, optimizer, training_data_loader):
    loss_meter = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        print 'Batch', iteration
        LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[0].cuda(async=True)), Variable(batch[1].cuda(async=True)), \
                                                    Variable( batch[2].cuda(async=True)), Variable(batch[3].cuda(async=True))

        HR_2, HR_4, HR_8 = model(LR)

        loss1 = criterion(HR_2, HR_2_target)
        loss2 = criterion(HR_4, HR_4_target)
        loss3 = criterion(HR_8, HR_8_target)
        loss = loss1 + loss2 + loss3
        loss_meter += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        print "===> Epoch {}, Avg. Loss: {:.4f}".format(epoch, loss_meter / float(iteration))


def test(model, criterion, testing_data_loader):
    avg_psnr1, avg_psnr2, avg_psnr3 = 0, 0, 0

    for batch in testing_data_loader:
        LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[0]), Variable(batch[1]), Variable(
            batch[2]), Variable(batch[3])

        LR = LR.cuda()
        HR_2_target = HR_2_target.cuda()
        HR_4_target = HR_4_target.cuda()
        HR_8_target = HR_8_target.cuda()

        HR_2, HR_4, HR_8 = model(LR)
        mse1 = criterion(HR_2, HR_2_target)
        mse2 = criterion(HR_4, HR_4_target)
        mse3 = criterion(HR_8, HR_8_target)
        psnr1 = 10 * log10(1 / mse1.data[0])
        psnr2 = 10 * log10(1 / mse2.data[0])
        psnr3 = 10 * log10(1 / mse3.data[0])
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
        avg_psnr3 += psnr3
    print "===> Avg. PSNR1: {:.4f} dB".format(avg_psnr1 / len(testing_data_loader))
    print "===> Avg. PSNR2: {:.4f} dB".format(avg_psnr2 / len(testing_data_loader))
    print "===> Avg. PSNR3: {:.4f} dB".format(avg_psnr3 / len(testing_data_loader))


def checkpoint(opt, epoch, model):

    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)
    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print "Checkpoint saved to {}".format(model_out_path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--train_dir', type=str, default='/home/tyantov/real_photo_12k_resized',
                        help='path to train dir')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.01')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.train_dir) #TODO: to param
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print '===> Building model'
    model = LasSRN()

    #criterion = CharbonnierLoss() #TODO: bugs to cuda
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    model = model.cuda()

    lr=opt.lr
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-5)
    print 'Starting learning'
    for epoch in range(1, opt.nEpochs + 1):
        print 'Epoch num=%d' % epoch
        train(epoch, model, criterion, optimizer, training_data_loader)
        test(model, criterion, testing_data_loader)
        if epoch:
            lr = lr/2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        checkpoint(opt, epoch, model)

if __name__ == '__main__':
    sys.exit(main())
