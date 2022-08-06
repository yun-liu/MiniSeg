import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.act(self.bn(self.conv(input)))
        return output


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


class DilatedParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DilatedParallelConvBlock, self).__init__()
        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))
        return output


class MiniSeg(nn.Module):
    def __init__(self, classes=2, P1=2, P2=3, P3=8, P4=6, aux=True):
        super(MiniSeg, self).__init__()
        self.aux = aux

        self.down1 = ConvBlock(3, 8, stride=2)
        self.level1 = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(8, 8))

        self.down2 = DilatedParallelConvBlock(8, 24, stride=2)
        self.level2 = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(DilatedParallelConvBlock(24, 24))

        self.down3 = DilatedParallelConvBlock(24, 32, stride=2)
        self.level3 = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(DilatedParallelConvBlock(32, 32))

        self.down4 = DilatedParallelConvBlock(32, 64, stride=2)
        self.level4 = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(DilatedParallelConvBlock(64, 64))

        self.up4_conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(64)
        self.up4_act = nn.PReLU(64)

        self.up3_conv4 = DilatedParallelConvBlockD2(64, 32)
        self.up3_conv3 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(32)
        self.up3_act = nn.PReLU(32)

        self.up2_conv3 = DilatedParallelConvBlockD2(32, 24)
        self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(24)
        self.up2_act = nn.PReLU(24)

        self.up1_conv2 = DilatedParallelConvBlockD2(24, 8)
        self.up1_conv1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(8)
        self.up1_act = nn.PReLU(8)

        if self.aux:
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(32, classes, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(8, classes, 1, stride=1, padding=0))

    def forward(self, input):
        output1 = self.down1(input)
        for layer in self.level1:
            output1 = layer(output1)

        output2 = self.down2(output1)
        for layer in self.level2:
            output2 = layer(output2)

        output3 = self.down3(output2)
        for layer in self.level3:
            output3 = layer(output3)

        output4 = self.down4(output3)
        for layer in self.level4:
            output4 = layer(output4)

        up4_conv4 = self.up4_bn4(self.up4_conv4(output4))
        up4 = self.up4_act(up4_conv4)

        up4 = F.interpolate(up4, output3.size()[2:], mode='bilinear', align_corners=False)
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

        if self.aux:
            pred4 = F.interpolate(self.pred4(up4), input.size()[2:], mode='bilinear', align_corners=False)
            pred3 = F.interpolate(self.pred3(up3), input.size()[2:], mode='bilinear', align_corners=False)
            pred2 = F.interpolate(self.pred2(up2), input.size()[2:], mode='bilinear', align_corners=False)
        pred1 = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)

        if self.aux:
            return (pred1, pred2, pred3, pred4,)
        else:
            return (pred1,)
