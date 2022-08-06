import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedParllelConvBlock(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super(DilatedParllelConvBlock, self).__init__()
        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=1, dilation=1, groups=nOut, bias=False)
        self.conv2 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=2, dilation=2, groups=nOut, bias=False)
        self.conv3 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=4, dilation=4, groups=nOut, bias=False)
        self.conv4 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=8, dilation=8, groups=nOut, bias=False)
        self.conv5 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=16, dilation=16, groups=nOut, bias=False)
        self.conv6 = nn.Conv2d(nOut, nOut, 3, stride=1, padding=32, dilation=32, groups=nOut, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        d5 = self.conv5(output)
        d6 = self.conv6(output)

        output = d1 + d2 + d3 + d4 + d5 + d6
        if self.add:
            output = input + output

        output = self.bn(output)
        output = self.act(output)

        return output

class DownsamplerBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownsamplerBlock, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            self.conv0 = nn.Conv2d(nIn, nOut-nIn, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut-nIn, nOut-nIn, 5, stride=2, padding=2, dilation=1, groups=nOut-nIn, bias=False)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut, nOut, 5, stride=2, padding=2, dilation=1, groups=nOut, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        if self.nIn < self.nOut:
            output = torch.cat([self.conv1(self.conv0(input)), self.pool(input)], 1)
        else:
            output = self.conv1(self.conv0(input))

        output = self.bn(output)
        output = self.act(output)

        return output

class NDC(nn.Module):
    '''
    This class defines the decoder network
    '''
    def __init__(self, classes=19, aux=True, P=3, Q=15):
        '''
        :param classes: number of classes in the dataset.
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super(NDC, self).__init__()
        self.level1 = DownsamplerBlock(3, 16)

        self.level2_0 = DownsamplerBlock(16, 60)
        self.level2 = nn.ModuleList()
        for i in range(0, P):
            self.level2.append(DilatedParllelConvBlock(60, 60))
        self.bn2 = nn.BatchNorm2d(120)
        self.act2 = nn.PReLU(120)

        self.level3_0 = DownsamplerBlock(120, 120)
        self.level3 = nn.ModuleList()
        for i in range(0, Q):
            self.level3.append(DilatedParllelConvBlock(120, 120))
        self.bn3 = nn.BatchNorm2d(240)
        self.act3 = nn.PReLU(240)

        self.classifier = nn.Conv2d(240, classes, 1, stride=1, padding=0)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output1 = self.level1(input)

        output2_0 = self.level2_0(output1)
        output2 = output2_0
        for layer in self.level2:
            output2 = layer(output2)
        output2 = self.bn2(torch.cat([output2_0, output2], 1))
        output2 = self.act2(output2)

        output3_0 = self.level3_0(output2)
        output3 = output3_0
        for layer in self.level3:
            output3 = layer(output3)
        output3 = self.bn3(torch.cat([output3_0, output3], 1))
        output3 = self.act3(output3)

        classifier = self.classifier(output3)
        classifier = F.interpolate(classifier, input.size()[2:], mode='bilinear', align_corners=False)

        return (classifier,)
