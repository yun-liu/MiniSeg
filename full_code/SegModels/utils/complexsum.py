# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch.nn as nn
import torchvision.models as tv_models

from utils.model_summary import model_summary


class CCustomNet(nn.Module):
    def __init__(self):
        super(CCustomNet, self).__init__()

        a = list()
        a.append(('conv_0', nn.Conv2d(3, 16, 3, 2, 1)))
        a.append(('bn_0', nn.BatchNorm2d(16)))
        a.append(('relu_0', nn.ReLU(inplace=True)))
        b = list()
        b.append(('conv_1', nn.Conv2d(16, 32, 3, 1, 1)))
        b.append(('bn_1', nn.BatchNorm2d(32)))
        b.append(('relu_1', nn.ReLU(inplace=True)))
        c = list()
        c.append(('conv_1', nn.Conv2d(32, 32, 3, 1, 1)))

        a = nn.Sequential(OrderedDict(a))
        b = nn.Sequential(OrderedDict(b))
        c = nn.Sequential(OrderedDict(c))
        self.d = nn.Sequential(OrderedDict(a=a, b=b, c=c))

    def forward(self, x):
        x = self.d(x)
        return x


def complexsum(model,input_size, device=-1):
    model_summary(model, input_size, query_granularity=2, device=device)
