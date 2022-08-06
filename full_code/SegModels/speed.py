import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import time

def computeTime(model, device='cuda'):
    inputs = torch.randn(30, 3, 512, 512)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): {:.4f}'.format(30*1/np.mean(time_spent)))


torch.backends.cudnn.benchmark = True

#from models import MiniSeg as net
#from models import UNet as net
#from models import NestedUNet as net
#from models import Att_UNet as net
#from models import FCN_VGG16 as net
#from models import CGNet as net
#from models import EDANet as net
#from models import ESPNet as net
#from models import ESPNetv2 as net
#from models import ENet as net
#from models import SegNet as net
#from models import NDC as net
#from models import FRRN as net
#from models import PSPNet as net
#from models import DFN as net
#from models import DenseASPP as net
#from models.DeepLabs import deeplabv3plus_xception as net
#from models.DeepLabs import deeplabv3 as net
#from models.baselines import ocnet as net
#from models.baselines import dunet as net
#from models.baselines import DANet as net
#from models.baselines import LEDNet as net
#from models.baselines import EncNet as net
#from models.baselines import BiSeNet as net
#from models import ShuffleNetV2 as net
#from models import MobileNetV2 as net
#from models.baselines import MobileNetv3 as net
#from models.baselines import EfficientNet as net
#from models.baselines import CCNet as net
#from models.baselines import ANN as net
#from models.baselines import GFF as net
#from models.baselines import InfNet as net

from models import MiniSeg_K2 as net

model = net.MiniSeg()


computeTime(model)
