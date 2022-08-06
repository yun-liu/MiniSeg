import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from models.baselines.segbase import SegBaseModel

__all__ = ['CCNet']

class CCNet(SegBaseModel):
    def __init__(self, classes=2, aux=True, backbone='resnet50', pretrained_base=True):
        super(CCNet, self).__init__(classes, aux, backbone, pretrained_base=pretrained_base)
        self.head = _CCHead(classes)

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return (x, )


class _CCHead(nn.Module):
    def __init__(self, classes=2, norm_layer=nn.BatchNorm2d):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer)
        self.out = nn.Conv2d(512, classes, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CC_module(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1))

    def forward(self, x, recurrence=1):
        out = self.conva(x)
        for i in range(recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out




#########################################
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CC_module(nn.Module):

    def __init__(self,in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x
