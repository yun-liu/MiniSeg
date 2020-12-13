#!/usr/bin/env python

import os
import cv2
import numpy as np
import os.path as osp
import SimpleITK as sitk
from PIL import Image


def compute_specificity(SEG, GT):
    TN = np.sum(np.logical_not(np.logical_or(SEG, GT)))
    FP = np.sum(SEG) - np.sum(np.logical_and(SEG, GT))
    spec = TN / (TN + FP)
    return spec


def evaluation_sample(SEG_np, GT_np):
    quality=dict()
    SEG_np = np.uint8(np.where(SEG_np, 1, 0))
    GT_np = np.uint8(np.where(GT_np, 1, 0))
    SEG = sitk.GetImageFromArray(SEG_np)
    GT = sitk.GetImageFromArray(GT_np)

    # Compute the evaluation criteria
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Overlap measures
    overlap_measures_filter.Execute(SEG, GT)
    #quality["jaccard"] = overlap_measures_filter.GetJaccardCoefficient()
    quality["dice"] = overlap_measures_filter.GetDiceCoefficient()
    quality["false_negative"] = overlap_measures_filter.GetFalseNegativeError()
    #quality["false_positive"] = overlap_measures_filter.GetFalsePositiveError()
    quality["sensitive"] = 1 - quality["false_negative"]
    quality["specificity"] = compute_specificity(SEG_np, GT_np)

    # Hausdorff distance
    hausdorff_distance_filter.Execute(SEG, GT)
    quality["hausdorff_distance"] = hausdorff_distance_filter.GetHausdorffDistance()

    return quality

#######################################################
#######################################################

def main():
    crossVal = 5
    avgmIOU = 0
    avgSEN = 0
    avgSPC = 0
    avgDSC = 0
    avgHD = 0
    data_root = '../datasets'
    data_name = 'CT100'
    model_name = 'MiniSeg'
    print(model_name, data_name)

    for i in range(crossVal):
        image_files = list()
        gt_files = list()
        pred_files = list()
        valFile = osp.join(data_root, 'COVID-19-' + data_name + '/dataList/'+'val'+str(i)+'.txt')
        pred_data_root = osp.join('./outputs/', data_name, model_name, 'crossVal'+str(i))
        with open(valFile) as text_file:
            for line in text_file:
                line_arr = line.split()
                image_files.append(osp.join(data_root, line_arr[0].strip()))
                gt_files.append(osp.join(data_root, line_arr[1].strip()))
                pred_files.append(osp.join(pred_data_root, line_arr[0].split('/')[-1].strip()))

        assert len(gt_files) == len(pred_files), 'The number of GT and pred must be equal'

        EPSILON = np.finfo(np.float).eps

        recall = np.zeros((len(gt_files)))
        precision = np.zeros((len(gt_files)))
        dice = np.zeros((len(gt_files)))
        sensitive = np.zeros((len(gt_files)))
        specificity = np.zeros((len(gt_files)))
        hausdorff_distance = np.zeros((len(gt_files)))

        mae = np.zeros((len(gt_files)))

        for idx in range(len(gt_files)):
            gt = cv2.imread(gt_files[idx], 0)
            pred = cv2.imread(pred_files[idx], 0)
            pred = pred.astype(np.float) / 255
            if not gt.shape == (512, 512):
                gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
            gt = gt == 255

            zeros = 0
            zeros_pred = []

            if np.max(pred) != 0:
                intersection = np.sum(np.logical_and(gt == pred, gt))
                recall[idx] = intersection * 1. / (np.sum(gt) + EPSILON)
                precision[idx] = intersection * 1. / (np.sum(pred) + EPSILON)
                dice[idx] = evaluation_sample(pred, gt).get("dice")
                sensitive[idx] = evaluation_sample(pred, gt).get("sensitive")
                specificity[idx] = evaluation_sample(pred, gt).get("specificity")
                hausdorff_distance[idx] = evaluation_sample(pred, gt).get("hausdorff_distance")
                #mae[idx] = np.sum(np.fabs(gt - pred)) * 1. / (gt.shape[0] * gt.shape[1])


        #recall = np.mean(recall, axis=0)
        #precision = np.mean(precision, axis=0)
        dice = np.max(np.mean(dice, axis=0))
        sensitive = np.max(np.mean(sensitive, axis=0))
        specificity = np.max(np.mean(specificity, axis=0))
        hausdorff_distance = np.max(np.mean(hausdorff_distance, axis=0))
        #F_beta = (1 + 0.3) * precision * recall / (0.3 * precision + recall + EPSILON)
        print(i,"{:.4f}".format(sensitive),"{:.4f}".format(specificity),"{:.4f}".format(dice),"{:.4f}".format(hausdorff_distance))

        avgSEN = avgSEN + sensitive/5
        avgSPC = avgSPC + specificity/5
        avgDSC = avgDSC + dice/5
        avgHD = avgHD + hausdorff_distance/5

    print("{:.4f}".format(avgSEN))
    print("{:.4f}".format(avgSPC))
    print("{:.4f}".format(avgDSC))
    print("{:.2f}".format(avgHD))

main()
