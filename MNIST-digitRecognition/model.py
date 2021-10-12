# -*- coding: utf-8 -*-
# @Time : 2021/10/7 19:58
# @Author : YangYu
# @Email: yangyu.cs@outlook.com
# @File : model.py
# @Software: PyCharm
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output
