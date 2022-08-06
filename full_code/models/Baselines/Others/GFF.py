import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.segbase import SegBaseModel

__all__ = ['GFF']


# The model may be wrong because lots of details missing in paper.
class GFF(SegBaseModel):
    def __init__(self, classes=2, aux=True, backbone='resnet50', pretrained_base=True):
        super(GFF, self).__init__(classes, aux, backbone, pretrained_base=pretrained_base)
        self.conv1_1 = nn.Conv2d(256, 256, 1, stride=2, padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
        self.conv3_1 = nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False)
        self.conv3_2 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
        self.conv4_1 = nn.Conv2d(2048, 256, 1, stride=1, padding=0, bias=False)
        self.conv4_2 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)

        self.x1gff = FuseGFFConvBlock(256, 256)
        self.x2gff = FuseGFFConvBlock(256, 256)
        self.x3gff = FuseGFFConvBlock(256, 256)
        self.x4gff = FuseGFFConvBlock(256, 256)

        self.psp = PyramidPoolingModule(2048)

        self.conv6 = nn.Conv2d(4096, 512, 3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.act = nn.ReLU(True)

        self.ConvUpscaleBlock1 = ConvUpscaleBlock(512, 256, kernel_size=3, stride=2)
        self.ConvBlock1 = ConvBlock(256, 256)
        self.ConvUpscaleBlock2 = ConvUpscaleBlock(256, 128, kernel_size=3, stride=2)
        self.ConvBlock2 = ConvBlock(128, 128)
        self.ConvUpscaleBlock3 = ConvUpscaleBlock(128, 64, kernel_size=3, stride=2)
        self.ConvBlock3 = ConvBlock(64, 64)

        self.conv7 = nn.Conv2d(64, classes, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        c1, c2, c3, c4 = self.base_forward(x)
        x1n = self.conv1_1(c1)
        g1 = self.conv1_2(x1n)
        g1 = torch.sigmoid(g1)
        x2n = self.conv2_1(c2)
        g2 = self.conv2_2(x2n)
        g2 = torch.sigmoid(g2)
        x3n = self.conv3_1(c3)
        g3 = self.conv3_2(x3n)
        g3 = torch.sigmoid(g3)
        x4n = self.conv4_1(c4)
        g4 = self.conv4_2(x4n)
        g4 = torch.sigmoid(g4)

        x1gff = (1+g1)*x1n + (1-g1)*(g2*x2n + g3*x3n + g4*x4n)
        x2gff = (1+g2)*x2n + (1-g2)*(g1*x1n + g3*x3n + g4*x4n)
        x3gff = (1+g3)*x3n + (1-g3)*(g2*x2n + g1*x1n + g4*x4n)
        x4gff = (1+g4)*x4n + (1-g4)*(g2*x2n + g3*x3n + g1*x1n)

        x1gff = self.x1gff(x1gff)
        x2gff = self.x2gff(x2gff)
        x3gff = self.x3gff(x3gff)
        x4gff = self.x4gff(x4gff)
        psp = self.psp(c4)

        x1gff = F.interpolate(x1gff, psp.size()[2:], mode='bilinear', align_corners=False)
        x2gff = F.interpolate(x2gff, psp.size()[2:], mode='bilinear', align_corners=False)
        x3gff = F.interpolate(x3gff, psp.size()[2:], mode='bilinear', align_corners=False)
        x4gff = F.interpolate(x4gff, psp.size()[2:], mode='bilinear', align_corners=False)

        d5 = torch.cat([psp, x1gff, x2gff, x3gff, x4gff], 1)
        d4 = torch.cat([x1gff, x2gff, x3gff, x4gff], 1)
        d3 = torch.cat([x1gff, x2gff, x3gff], 1)
        d2 = torch.cat([x1gff, x2gff], 1)
        d1 = torch.cat([x1gff], 1)

        full_block = torch.cat([d1, d2, d3, d4, d5], 1)

        out = self.act(self.bn6(self.conv6(full_block)))

        out = self.ConvBlock1(self.ConvUpscaleBlock1(out))
        out = self.ConvBlock2(self.ConvUpscaleBlock2(out))
        out = self.ConvBlock3(self.ConvUpscaleBlock3(out))

        out = self.conv7(out)
        pred1 = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=False)

        return (pred1, )

class PyramidPoolingModule(nn.Module):
    """Ref: Pyramid Scene Parsing Network,CVPR2017, http://arxiv.org/abs/1612.01105 """
    def __init__(self, inChannel, midReduction=4, outChannel=512, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.midChannel= int(inChannel/midReduction)  #1x1Conv channel num, defalut=512
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(inChannel, self.midChannel, size) for size in sizes])  #pooling->conv1x1
        self.bottleneck = nn.Conv2d( (inChannel+ self.midChannel*4), outChannel, kernel_size=3)  #channel: 4096->512 1x1
        self.bn = nn.BatchNorm2d(outChannel)
        self.prelu = nn.PReLU()

    def _make_stage(self, inChannel, midChannel,  size):
        pooling = nn.AdaptiveAvgPool2d(output_size=(size, size))
        Conv = nn.Conv2d(inChannel, midChannel, kernel_size=1, bias=False)
        return nn.Sequential(pooling, Conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        mulBranches = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]  #four parallel baranches
        out = self.bottleneck( torch.cat( (mulBranches[0], mulBranches[1], mulBranches[2], mulBranches[3],feats) ,1))
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        if kernel_size==3:
            padding = 1
        if kernel_size==7:
            padding = 3
        self.bn = nn.BatchNorm2d(in_planes)
        self.act = nn.ReLU(True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, input):
        output = self.conv(self.act(self.bn(input)))
        return output

class FuseGFFConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(FuseGFFConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.act(self.bn1(self.conv1(input)))
        output = self.act(self.bn2(self.conv2(output)))
        return output


class ConvUpscaleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=2):
        super(ConvUpscaleBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.act = nn.ReLU(True)
        self.deconv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, padding=1, bias=False)

    def forward(self, input):
        output = self.deconv(self.act(self.bn(input)))
        return output
