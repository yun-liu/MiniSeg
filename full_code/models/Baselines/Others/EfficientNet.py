"""MobileNetV1 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.base import BaseModel


class EfficientNet(BaseModel):
    def __init__(self, classes=2, aux=False, backbone='efficientnet', pretrained_base=False, **kwargs):
        super(EfficientNet, self).__init__(classes, aux, backbone, pretrained_base, **kwargs)

        self.up5_conv6 = DilatedParallelConvBlockD2(320, 192)
        self.up5_conv5 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.up5_bn5 = nn.BatchNorm2d(192)
        self.up5_act = nn.PReLU(192)

        self.up4_conv5 = DilatedParallelConvBlockD2(192, 112)
        self.up4_conv4 = nn.Conv2d(112, 112, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(112)
        self.up4_act = nn.PReLU(112)

        self.up3_conv4 = DilatedParallelConvBlockD2(112, 80)
        self.up3_conv3 = nn.Conv2d(80, 80, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(80)
        self.up3_act = nn.PReLU(80)

        self.up2_conv3 = DilatedParallelConvBlockD2(80, 24)
        self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(24)
        self.up2_act = nn.PReLU(24)

        self.up1_conv2 = DilatedParallelConvBlockD2(24, 16)
        self.up1_conv1 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(16)
        self.up1_act = nn.PReLU(16)

        self.pred1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(16, classes, 1, stride=1, padding=0))

    def forward(self, x):
        size = x.size()[2:]
        output1, output2, output3, output4, output5, output6 = self.base_forward(x)

        up6 = F.interpolate(output6, output5.size()[2:], mode='bilinear', align_corners=False)
        up5_conv6 = self.up5_conv6(up6)
        up5_conv5 = self.up5_bn5(self.up5_conv5(output5))
        up5 = self.up5_act(up5_conv6 + up5_conv5)

        up5 = F.interpolate(up5, output4.size()[2:], mode='bilinear', align_corners=False)
        up4_conv5 = self.up4_conv5(up5)
        up4_conv4 = self.up4_bn4(self.up4_conv4(output4))
        up4 = self.up4_act(up4_conv5 + up4_conv4)

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

        pred1 = F.interpolate(self.pred1(up1), x.size()[2:], mode='bilinear', align_corners=False)

        return (pred1,)


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
