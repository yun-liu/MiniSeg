import torch.nn as nn

from models.baselines.base_model import mobilenet1_0, mobilenet_v2_1_0, mobilenet_v3_small_1_0, mobilenet_v3_large_1_0, igcv3_1_0, efficientnet_b0


class BaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenet', pretrained_base=False, **kwargs):
        super(BaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.backbone = backbone

        if backbone == 'mobilenet':
            self.pretrained = mobilenet1_0()
        elif backbone == 'mobilenetv2':
            self.pretrained = mobilenet_v2_1_0()
        elif backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0()
        elif backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0()
        elif backbone == 'igcv3':
            self.pretrained = igcv3_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'efficientnet':
            self.pretrained = efficientnet_b0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError("Unknown backnone: {}".format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone in ['igcv3', 'efficientnet']:
            x = self.pretrained.conv1(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
            c5 = self.pretrained.layer5(c4)
            if self.backbone == 'efficientnet':
                c6 = self.pretrained.layer6(c5)
                return c1, c2, c3, c4, c5, c6
            return c1, c2, c3, c4, c5
        elif self.backbone in ['mobilenet', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large']:
            x = self.pretrained.conv1(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
            c5 = self.pretrained.layer5(c4)
            return c1, c2, c3, c4, c5
        else:
            raise ValueError
