import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FastSal(nn.Module):
    def __init__(self, classes=2, aux=False, pretrained='ShuffleNetV2_pretrain.pth'):
        super(FastSal, self).__init__()
        self.context_path = ShuffleNetV2(pretrained)
        if pretrained is not None:
            self.context_path.load_state_dict(torch.load(pretrained))

        self.up3_conv4 = DilatedParallelConvBlockD2(1024, 232)
        self.up3_conv3 = nn.Conv2d(232, 232, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(232)
        self.up3_act = nn.PReLU(232)

        self.up2_conv3 = DilatedParallelConvBlockD2(232, 116)
        self.up2_conv2 = nn.Conv2d(116, 116, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(116)
        self.up2_act = nn.PReLU(116)

        self.up1_conv2 = DilatedParallelConvBlockD2(116, 24)
        self.up1_conv1 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(24)
        self.up1_act = nn.PReLU(24)

        self.pred1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))

    def forward(self, x): # (3, 1)
        output1, output2, output3, output4 = self.context_path(x)
        # (24, 1/4) (116, 1/8) (232, 1/16)  (1024, 1/32)

        up4 = F.interpolate(output4, output3.size()[2:], mode='bilinear', align_corners=False)
        up3_conv4 = self.up3_conv4(up4)
        up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        up3 = self.up3_act(up3_conv4 + up3_conv3)

        up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        up2_conv3 = self.up2_conv3(up3)
        up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        up2 = self.up2_act(up2_conv3 + up2_conv2)

        up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        up1_conv2 = self.up1_conv2(up2)
        up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        up1 = self.up1_act(up1_conv2 + up1_conv1)

        pred1 = F.interpolate(self.pred1(up1), x.size()[2:], mode='bilinear', align_corners=False)

        return (pred1,)

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output
        
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )
    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )
    def forward(self, x):
        return self.conv(x)


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2

        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, pretrained=None):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = conv_bn(3, 24, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = nn.Sequential(
                InvertedResidual(24, 116, 2, 2),
                InvertedResidual(116, 116, 1, 1),
                InvertedResidual(116, 116, 1, 1),
                InvertedResidual(116, 116, 1, 1)
                )

        self.stage2 = nn.Sequential(
                InvertedResidual(116, 232, 2, 2),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1),
                InvertedResidual(232, 232, 1, 1)
                )

        self.stage3 = nn.Sequential(
                InvertedResidual(232, 464, 2, 2),
                InvertedResidual(464, 464, 1, 1),
                InvertedResidual(464, 464, 1, 1),
                InvertedResidual(464, 464, 1, 1)
                )

        self.conv_last = conv_1x1_bn(464, 1024)

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained model loaded!')


    def forward(self, x):
        out1 = self.maxpool(self.conv1(x))
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out4 = self.conv_last(self.stage3(out3))
        return out1, out2, out3, out4
