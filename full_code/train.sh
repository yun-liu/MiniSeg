#!/bin/bash
# --max_epochs = [50, 60, 80, 100]
# --lr = [5e-4, 1e-4, 1e-3, 5e-3]
# --batch_size = [4, 5, 6, 8]
# --weight_decay = [5e-5, 1e-5, 5e-4, 1e-4]
# --random_seed = [-1, 0, 1, 42, 100] # if random_seed=-1, random_seed is unfixed
# --data_name = [CT100, P9, P20, P1110]

CUDA_VISIBLE_DEVICES=0 python train.py --max_epochs 80 \
                                         --lr 1e-4 \
                                         --batch_size 5 \
                                         --weight_decay 1e-4 \
                                         --random_seed -1 \
                                         --model_name MiniSeg \
                                         --data_name CT100 \
                                         --savedir ./results_MiniSeg_crossVal