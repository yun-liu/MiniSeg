import sys
sys.path.insert(0, '.')
import argparse
import torch
import os
from utils.simplesum import simplesum
from utils.complexsum import complexsum
from utils.valsum import valsum
from utils.runtime import runtime

###### add your model ######
from models import MiniSeg_K2 as net
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
#from models import MiniSeg as net

model = net.MiniSeg()

parser = argparse.ArgumentParser(description='PyTorch Summary')
parser.add_argument('--mod', default='simple', type=str,
                    help='simple complex val')
parser.add_argument('--gpu', default='0',type=int,
                    help='GPU: ID or CPU: -1')
parser.add_argument('--size', default="3,512,512", type=str,
                    help='Size of input image (C,H,W)')
parser.add_argument('--runtime', default='200',type=int,
                    help='-1: not enable runtime test. runtime>=1 average iters for runtime test.')
args = parser.parse_args()

if args.gpu >= 0:
  if torch.cuda.is_available():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Using GPU:",args.gpu)
    device = torch.device('cuda', args.gpu)
    device_mode = 'gpu'
  else:
    print("CUDA is not available.")
    device = torch.device('cpu')
    device_mode = 'cpu'
else:
  print("Using CPU")
  device = torch.device('cpu')
  device_mode = 'cpu'

model = model.to(device)
inputsize = args.size.split(',')
inputsize = [int(x) for x in inputsize]
print("Input size:", inputsize)
if args.mod =='simple':
    print("Using SIMPLE summary mode:")
    model.eval()
    simplesum(model, inputsize, device=args.gpu)
elif args.mod =='complex':
    print("Using COMPLEX summary mode:")
    complexsum(model, inputsize, device=args.gpu)
elif args.mod =='val':
    print("Using VALIDATION summary mode: (Only support GPU mode.)")
    valsum(model, inputsize, device=args.gpu)
else:
    print("Only support simple|complex|val modes.")

#if args.runtime > 0:
#    runtime(model, inputsize, iter=args.runtime, device=args.gpu)
