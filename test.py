# coding:utf-8

import torch
import torchvision as tv

import model
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import DefaultConfig


def data_loader(root, batch_size, num_workers):
    print("Test dataset is downloading. Please Wait...")

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return testloader


def test(model_test, dataloader, opt):
    if opt.use_gpu:
        print("Test started with CUDA")
        model_test.cuda()
    else:
        print("Test started without CUDA")

        model_test.eval()
    total_num = 0
    correct_num = 0

    for i, (data, label) in enumerate(dataloader):
        if opt.use_gpu:
            data = data.cuda()

        score = model_test(data)

        _, predict = torch.max(score.data, 1)

        total_num += label.size(0)
        correct_num += (predict.cpu() == label).sum()

    return 100 * float(correct_num) / float(total_num)


if __name__ == '__main__':
    print("Initialize starting options")
    opt = DefaultConfig()

    opt.parse({'batch_size': 128, 'num_workers': 4})

    model_test = getattr(model, opt.model)()
    model_test.load("model/SuleymanNET_model_state_dict.pkl")

    testloader = data_loader(opt.root, opt.batch_size, opt.num_workers)

    accuracy = test(model_test, testloader, opt)

    print("Accuracy of Test Set: %.3f" % accuracy)
