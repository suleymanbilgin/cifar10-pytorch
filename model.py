# coding:utf-8

import torch
import time
from torch import nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))


class SuleymanNet(BasicModule):
    def __init__(self):
        # the input size: (batch, 3, 32, 32)
        super(SuleymanNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                                      nn.ELU(True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(32, 45, 5),
                                      nn.ELU(True),
                                      nn.Conv2d(45, 64, 5),
                                      nn.ELU(True)
                                      )

        self.classifier = nn.Sequential(nn.Linear(4096, 1600),
                                        nn.ELU(True),
                                        nn.Linear(1600, 120),
                                        nn.ELU(True),
                                        nn.Linear(120, 84),
                                        nn.ELU(True),
                                        nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    data = torch.randn((128, 3, 32, 32))
