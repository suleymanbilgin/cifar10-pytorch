# coding:utf-8

import torch
import torchvision as tv
from tqdm import tqdm

import model
import averagevaluemeter
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import DefaultConfig


def data_loader(root, batch_size, num_workers):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = tv.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader


def train(opt):
    model_train = getattr(model, opt.model)()

    if opt.use_gpu:
        model_train.cuda()

    trainloader = data_loader(opt.root, opt.batch_size, opt.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.SGD(model_train.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # meter
    loss_meter = averagevaluemeter.AverageValueMeter()
    previous_loss = 1e100

    for epoch in range(opt.max_epoch):
        print('Epoch: %d' % (epoch))

        loss_meter.reset()

        for ii, (data, label) in tqdm(enumerate(trainloader)):
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            score = model_train(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

        model_train.train()

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param in optimizer.param_groups:
                param['lr'] = lr
                print("Changing learning rate to %.19f" % lr)

        previous_loss = loss_meter.value()[0]

    torch.save(model_train.state_dict(), "model/SuleymanNET_model_state_dict.pkl")


if __name__ == '__main__':
    opt = DefaultConfig()
    opt.parse({'max_epoch': 110, 'weight_decay': 15e-4, 'lr': 0.1})

    train(opt)
