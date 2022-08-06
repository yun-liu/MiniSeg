#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxtpku@pku.edu.cn)
# Implementation of Paper Learning a Discriminative Feature Network for Semantic Segmentation (CVPR2018)(face_plus_plus)
import torch
import torch.nn as nn
from torchvision.models import resnet50
#import torch.utils.model_zoo as model_zoo
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
'''
__all__ = ["DFN"]

class DFN(nn.Module):
    def __init__(self, classes=2, aux=True):
        super(DFN, self).__init__()
        self.classes = classes
        self.resnet_features = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1,
                                    self.resnet_features.relu  #self.resnet_features.conv3,
                                    #self.resnet_features.bn3, self.resnet_features.relu3
                                    )
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        # this is for smooth network
        self.out_conv = nn.Conv2d(2048,self.classes,kernel_size=1,stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cab1 = CAB(self.classes*2,self.classes)
        self.cab2 = CAB(self.classes*2,self.classes)
        self.cab3 = CAB(self.classes*2,self.classes)
        self.cab4 = CAB(self.classes*2,self.classes)

        self.rrb_d_1 = RRB(256, self.classes)
        self.rrb_d_2 = RRB(512, self.classes)
        self.rrb_d_3 = RRB(1024, self.classes)
        self.rrb_d_4 = RRB(2048, self.classes)

        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear")
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear")

        self.rrb_u_1 = RRB(self.classes,self.classes)
        self.rrb_u_2 = RRB(self.classes,self.classes)
        self.rrb_u_3 = RRB(self.classes,self.classes)
        self.rrb_u_4 = RRB(self.classes,self.classes)


        ## this is for boarder net work
        self.rrb_db_1 = RRB(256, self.classes)
        self.rrb_db_2 = RRB(512, self.classes)
        self.rrb_db_3 = RRB(1024, self.classes)
        self.rrb_db_4 = RRB(2048, self.classes)

        self.rrb_trans_1 = RRB(self.classes,self.classes)
        self.rrb_trans_2 = RRB(self.classes,self.classes)
        self.rrb_trans_3 = RRB(self.classes,self.classes)

        self.res_upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.res1_upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")


    def forward(self, x):
        # suppose input = x , if x 512
        f0 = self.layer0(x)  # 256
        f1 = self.layer1(f0)  # 128
        f2 = self.layer2(f1)  # 64
        f3 = self.layer3(f2)  # 32
        f4 = self.layer4(f3)  # 16

        # for border network
        res1 = self.rrb_db_1(f1)
        res1 = self.rrb_trans_1(res1 + self.upsample(self.rrb_db_2(f2)))
        res1 = self.rrb_trans_2(res1 + self.upsample_4(self.rrb_db_3(f3)))
        res1 = self.rrb_trans_3(res1 + self.upsample_8(self.rrb_db_4(f4)))
        # print (res1.size())
        # for smooth network
        res2 = self.out_conv(f4)
        res2 = self.global_pool(res2)  #
        res2 = nn.Upsample(size=f4.size()[2:],mode="nearest")(res2)

        f4 = self.rrb_d_4(f4)
        res2 = self.cab4([res2,f4])
        res2 = self.rrb_u_1(res2)

        f3 = self.rrb_d_3(f3)
        res2 = self.cab3([self.upsample(res2),f3])
        res2 =self.rrb_u_2(res2)

        f2 = self.rrb_d_2(f2)
        res2 = self.cab2([self.upsample(res2), f2])
        res2 =self.rrb_u_3(res2)

        f1 = self.rrb_d_1(f1)
        res2 = self.cab1([self.upsample(res2), f1])
        res2 = self.rrb_u_4(res2)

        res1 = self.res_upsample_4(res1)
        res2 = self.res1_upsample_4(res2)

        return (res2, res1, )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


'''
def resnet101(pretrained=False):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3])
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model



class ResNetV1b(nn.Module):

    def __init__(self, block, layers, num_classes=2, dilated=True, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
'''
