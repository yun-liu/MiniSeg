"""Base Model for Semantic Segmentation"""
import torch.nn as nn
from models.baselines.resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    def __init__(self, classes=2, aux=False, backbone='resnet50', pretrained_base=True):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.classes = classes
        self.backbone = backbone
        if self.backbone == 'resnet50':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101_v1b(pretrained=pretrained_base)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152_v1b(pretrained=pretrained_base)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
