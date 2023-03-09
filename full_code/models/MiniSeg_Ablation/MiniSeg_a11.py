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
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)


    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        p = self.pool(output)
        d1 = d1 + p
        d2 = d1 + d2
        d3 = d2 + d3
        d4 = d3 + d4
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))

        return output


class DownsamplerBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(DownsamplerBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 5, stride=stride, padding=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1(self.conv0(input))
        output = self.act(self.bn(output))
        return output



class MiniSeg(nn.Module):
    def __init__(self, classes=2, P1=2, P2=3, P3=8, P4=6, aux=True):
        super(MiniSeg, self).__init__()
        self.D1 = int(P1/2)
        self.D2 = int(P2/2)
        self.D3 = int(P3/2)
        self.D4 = int(P4/2)
        self.aux = aux

        self.long1 = DownsamplerBlock(3, 16, stride=2)
        self.down1 = ConvBlock(3, 16, stride=2)
        self.level1 = nn.ModuleList()
        self.level1_long = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(16, 16))
        for i in range(0, self.D1):
            self.level1_long.append(DownsamplerBlock(16, 16, stride=1))

        self.long2 = DownsamplerBlock(16, 48, stride=2)
        self.down2 = DilatedParallelConvBlock(16, 48, stride=2)
        self.level2 = nn.ModuleList()
        self.level2_long = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(DilatedParallelConvBlock(48, 48))
        for i in range(0, self.D2):
            self.level2_long.append(DownsamplerBlock(48, 48, stride=1))

        self.long3 = DownsamplerBlock(48, 64, stride=2)
        self.down3 = DilatedParallelConvBlock(48, 64, stride=2)
        self.level3 = nn.ModuleList()
        self.level3_long = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(DilatedParallelConvBlock(64, 64))
        for i in range(0, self.D3):
            self.level3_long.append(DownsamplerBlock(64, 64, stride=1))

        self.long4 = DownsamplerBlock(64, 128, stride=2)
        self.down4 = DilatedParallelConvBlock(64, 128, stride=2)
        self.level4 = nn.ModuleList()
        self.level4_long = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(DilatedParallelConvBlock(128, 128))
        for i in range(0, self.D4):
            self.level4_long.append(DownsamplerBlock(128, 128, stride=1))

        self.up4_conv4 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(128)
        self.up4_act = nn.PReLU(128)

        self.up3_conv4 = DilatedParallelConvBlockD2(128, 64)
        self.up3_conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(64)
        self.up3_act = nn.PReLU(64)

        self.up2_conv3 = DilatedParallelConvBlockD2(64, 48)
        self.up2_conv2 = nn.Conv2d(48, 48, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(48)
        self.up2_act = nn.PReLU(48)

        self.up1_conv2 = DilatedParallelConvBlockD2(48, 16)
        self.up1_conv1 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(16)
        self.up1_act = nn.PReLU(16)

        if self.aux:
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(128, classes, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(48, classes, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(16, classes, 1, stride=1, padding=0))

    def forward(self, input):
        long1 = self.long1(input)
        output1 = self.down1(input)
        output1_add = output1 + long1
        for i, layer in enumerate(self.level1):
            if i < self.D1:
                output1 = layer(output1_add) + output1
                long1 = self.level1_long[i](output1_add) + long1
                output1_add = output1 + long1
            else:
                output1 = layer(output1_add) + output1
                output1_add = output1 + long1

        long2 = self.long2(output1 + long1)
        output2 = self.down2(output1)
        output2_add = output2 + long2
        for i, layer in enumerate(self.level2):
            if i < self.D2:
                output2 = layer(output2_add) + output2
                long2 = self.level2_long[i](output2_add) + long2
                output2_add = output2 + long2
            else:
                output2 = layer(output2_add) + output2
                output2_add = output2 + long2

        long3 = self.long3(output2 + long2)
        output3 = self.down3(output2)
        output3_add = output3 + long3
        for i, layer in enumerate(self.level3):
            if i < self.D3:
                output3 = layer(output3_add) + output3
                long3 = self.level3_long[i](output3_add) + long3
                output3_add = output3 + long3
            else:
                output3 = layer(output3_add) + output3
                output3_add = output3 + long3

        long4 = self.long4(output3 + long3)
        output4 = self.down4(output3)
        output4_add = output4 + long4
        for i, layer in enumerate(self.level4):
            if i < self.D4:
                output4 = layer(output4_add) + output4
                long4 = self.level4_long[i](output4_add) + long4
                output4_add = output4 + long4
            else:
                output4 = layer(output4_add) + output4
                output4_add = output4 + long4

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
            return (pred1, pred2, pred3, pred4, )
        else:
            return (pred1, )