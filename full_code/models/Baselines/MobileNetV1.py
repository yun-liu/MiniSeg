import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSal(nn.Module):
    def __init__(self, classes=2, aux=False, pretrained='MobileNetV1_pretrain.pth'):
        super(FastSal, self).__init__()
        self.context_path = MobileNetV1(pretrained)
        if pretrained is not None:
            self.context_path.load_state_dict(torch.load(pretrained))

        self.up4_conv5 = DilatedParallelConvBlockD2(1024, 512)
        self.up4_conv4 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(512)
        self.up4_act = nn.PReLU(512)

        self.up3_conv4 = DilatedParallelConvBlockD2(512, 256)
        self.up3_conv3 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(256)
        self.up3_act = nn.PReLU(256)

        self.up2_conv3 = DilatedParallelConvBlockD2(256, 128)
        self.up2_conv2 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(128)
        self.up2_act = nn.PReLU(128)

        self.up1_conv2 = DilatedParallelConvBlockD2(128, 64)
        self.up1_conv1 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(64)
        self.up1_act = nn.PReLU(64)

        self.pred1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))


    def forward(self, x): # (3, 1)
        output1, output2, output3, output4, output5 = self.context_path(x)
        # (64, 1/2) (128, 1/4) (256, 1/8)  (512, 1/16) (1024, 1/32)

        up5 = F.interpolate(output5, output4.size()[2:], mode='bilinear', align_corners=False)
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


class MobileNetV1(nn.Module):
    def __init__(self, pretrained=None):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.stage1 = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1)
        )
        self.stage2 = nn.Sequential(
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1)
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        self.stage4 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1)
        )
        self.stage5 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained model loaded!')

    def forward(self, x):
        out1 = self.stage1(x)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        out5 = self.stage5(out4)

        return out1, out2, out3, out4, out5
