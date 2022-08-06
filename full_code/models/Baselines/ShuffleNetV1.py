import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FastSal(nn.Module):
    def __init__(self, classes=2, aux=False, pretrained='ShuffleNetV1_pretrain.pth'):
        super(FastSal, self).__init__()
        self.context_path = ShuffleNetV1(pretrained)
        if pretrained is not None:
            self.context_path.load_state_dict(torch.load(pretrained))

        self.up3_conv4 = DilatedParallelConvBlockD2(960, 480)
        self.up3_conv3 = nn.Conv2d(480, 480, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(480)
        self.up3_act = nn.PReLU(480)

        self.up2_conv3 = DilatedParallelConvBlockD2(480, 240)
        self.up2_conv2 = nn.Conv2d(240, 240, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(240)
        self.up2_act = nn.PReLU(240)

        self.up1_conv2 = DilatedParallelConvBlockD2(240, 24)
        self.up1_conv1 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(24)
        self.up1_act = nn.PReLU(24)

        self.pred1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))

    def forward(self, x): # (3, 1)
        output1, output2, output3, output4 = self.context_path(x)
        # (24, 1/4) (240, 1/8) (480, 1/16)  (960, 1/32)
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


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    '''3x3 convolution with padding
    '''
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    '''1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    '''
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):

        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError('Cannot combine tensors with \'{}\'' \
                             'Only \'add\' and \'concat\' are' \
                             'supported'.format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
            )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False
            )

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):
        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3,
                stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNetV1(nn.Module):
    '''ShuffleNetV1 implementation.
    '''
    def __init__(self, pretrained=None):
        '''ShuffleNetV1 constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        '''
        super(ShuffleNetV1, self).__init__()

        self.groups = 3
        self.stage_repeats = [3, 7, 3]
        self.in_channels =  3
        self.num_classes = 2

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        self.stage_out_channels = [-1, 24, 240, 480, 960]

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1], # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained model loaded!')

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
            )
        modules[stage_name+'_0'] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + '_{}'.format(i+1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
                )
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        out1 = self.maxpool(x)

        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)

        return out1, out2, out3, out4
