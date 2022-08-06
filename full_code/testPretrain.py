import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from IOUEval import iouEval
from PIL import Image
import loadData as ld
#from models import MiniSegV40 as net
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
from models import DFN as net
#from models import DenseASPP as net
#from models.DeepLabs import deeplabv3plus_xception as net
#from models.DeepLabs import deeplabv3 as net
#from models.baselines import ocnet as net
#from models.baselines import dunet as net
#from models.baselines import DANet as net
#from models.baselines import LEDNet as net
#from models.baselines import EncNet as net
#from models.baselines import BiSeNet as net
#from models import ShuffleNetV1 as net
#from models import MobileNetV2 as net
#from models.baselines import MobileNetv3 as net
#from models.baselines import EfficientNet as net
#from models.baselines import CCNet as net
#from models.baselines import GFF as net
#from models.baselines import ANN as net


@torch.no_grad()
def validate(args, model, image_list, label_list, crossVal, mean, std):
    iou_eval_val = iouEval(args.classes)
    for idx in range(len(image_list)):
        image = cv2.imread(image_list[idx]) / 255
        label = cv2.imread(label_list[idx], 0) / 255
        #label = np.array(Image.open(label_list[idx]))

        img = image.astype(np.float32)
        img -= mean
        img /= std

        img = cv2.resize(img, (args.width, args.height))
        img = img.transpose((2, 0, 1))
        img_variable = Variable(torch.from_numpy(img).unsqueeze(0))
        if args.gpu:
            img_variable = img_variable.cuda()

        start_time = time.time()
        img_out = model(img_variable)[0]

        torch.cuda.synchronize()
        diff_time = time.time() - start_time
        print('Segmentation for {}/{} takes {:.3f}s per image'.format(idx, len(image_list), diff_time))

        #img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        class_numpy = img_out[0].max(0)[1].data.cpu().numpy()
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        iou_eval_val.add_batch(class_numpy, label)

        out_numpy = (class_numpy * 255).astype(np.uint8)
        name = image_list[idx].split('/')[-1]
        if not osp.isdir(osp.join(args.savedir, args.data_name)):
            os.mkdir(osp.join(args.savedir, args.data_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name)):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal))):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal)))
        cv2.imwrite(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal), name[:-4] + '.png'), out_numpy)

    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()
    print('Overall Acc (Val): %.4f\t mIOU (Val): %.4f' % (overall_acc, mIOU))
    return mIOU


def main_te(args, crossVal, pretrained, mean, std):
    # read the image list
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, 'COVID-19-' + args.data_name + '/dataList/'+'val'+str(crossVal)+'.txt')) as text_file:
        for line in text_file:
            line_arr = line.split()
            image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
            label_list.append(osp.join(args.data_dir, line_arr[1].strip()))

    model = net.DFN(args.classes, aux=False)
    if not osp.isfile(pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(pretrained)
    #'''
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        if 'pred' not in key or 'pred1' in key:
            new_keys.append(key) # .replace('module.', '')
            new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict)
    #'''
    model.load_state_dict(state_dict, True)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.mkdir(args.savedir)

    mIOU = validate(args, model, image_list, label_list, crossVal, mean, std)

    return mIOU


def main(args):
    crossVal = 5
    maxEpoch = [80, 80, 80, 80, 80]
    mIOUList = []
    avgmIOU = 0

    for i in range(crossVal):
        dataLoad = ld.LoadData(args.data_dir, args.classes)
        data = dataLoad.processData(i, args.data_name)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)#RGB
        print('Data statistics:')
        print(mean, std)

        pthName = 'model_' + args.model_name + '_crossVal' + str(i+1) + '_' + str(maxEpoch[i]) + '.pth'
        pretrainedModel = args.pretrained + args.data_name + '/' + args.model_name + '/' + pthName
        mIOU = "{:.4f}".format(main_te(args, i, pretrainedModel, mean, std))
        mIOU = float(mIOU)
        mIOUList.append(mIOU)
        avgmIOU = avgmIOU + mIOU/5
    print(mIOUList)
    print(args.model_name, args.data_name, "{:.4f}".format(avgmIOU))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="../datasets", help='Data directory')
    parser.add_argument('--width', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--savedir', default='./new_outputs', help='directory to save the results')
    parser.add_argument('--model_name', default='DFN', help='Model name')
    parser.add_argument('--data_name', default='P20', help='Model name')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default='./new_results_MiniSeg_crossVal_mod80/', help='Pretrained model')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset')

    args = parser.parse_args()
    print('Called with args:')
    print(args)
    main(args)
