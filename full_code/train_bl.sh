#!/bin/bash
# --max_epochs = [40, 50, 60, 70, 80, 90, 100]
# --lr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
# --batch_size = [4, 5, 6, 8, 12, 16]
# --weight_decay = [1e-5, 5e-5, 1e-4, 5e-4]
# --random_seed = [-1, 0, 1, 42, 100] # if random_seed=-1, random_seed is unfixed
# --data_name = [CT100, P9, P20, P1110]

# python trainPretrain.py
# --model_name =
#       [FCN, PSPNet, DFN, DeeplabV3, DeeplabV3P, OCNet, DUNet, DANet, EncNet,
#       BiSeNet, ShuffleNet, MobileNet, EfficientNet, ANNNet, CCNet, GFF]

# python train_bl.py
# --model_name =
#       [U_Net, NestedUNet, Att_UNet, ESPNet, EDANet, ESPNet, ESPNetv2,
#       ENet, CGNet, SegNet, FRRN, DenseASPP, LEDNet]

# python trainInfNet.py
# --model_name = InfNet


CUDA_VISIBLE_DEVICES=0 python trainPretrain.py --max_epochs 80 \
                                         --lr 1e-4 \
                                         --batch_size 5 \
                                         --weight_decay 1e-4 \
                                         --random_seed -1 \
                                         --model_name FCN \
                                         --data_name CT100 \
                                         --savedir ./results_MiniSeg_crossVal