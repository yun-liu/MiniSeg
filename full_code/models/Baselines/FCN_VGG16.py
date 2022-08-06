import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, classes=2, aux=True, pretrained='vgg16_bn-6c64b313.pth'):
        super(FCN, self).__init__()
        self.vgg16 = VGG16BN()
        if pretrained is not None:
            self.vgg16.load_state_dict(torch.load(pretrained), strict=True)

        self.FCN_head = nn.Sequential(ConvBNReLU(512, 256),
                                      nn.Dropout2d(0.1, False),
                                      nn.Conv2d(256, classes, 1, stride=1, padding=0))

    def forward(self, input):
        _, _, _, _, conv5 = self.vgg16(input)
        pred = self.FCN_head(conv5)
        pred = F.interpolate(pred, input.size()[2:], mode='bilinear', align_corners=False)

        return (pred,)


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class VGG16BN(nn.Module):
    def __init__(self):
        super(VGG16BN, self).__init__()
        self.conv1_1 = ConvBNReLU(3, 64)
        self.conv1_2 = ConvBNReLU(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNReLU(64, 128)
        self.conv2_2 = ConvBNReLU(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBNReLU(128, 256)
        self.conv3_2 = ConvBNReLU(256, 256)
        self.conv3_3 = ConvBNReLU(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBNReLU(256, 512)
        self.conv4_2 = ConvBNReLU(512, 512)
        self.conv4_3 = ConvBNReLU(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv5_1 = ConvBNReLU(512, 512, dilation=2, pad=2)
        self.conv5_2 = ConvBNReLU(512, 512, dilation=2, pad=2)
        self.conv5_3 = ConvBNReLU(512, 512, dilation=2, pad=2)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
