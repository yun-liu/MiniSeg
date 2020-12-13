import torch
import numpy as np

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class iouEval(object):
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.nClasses, self.nClasses))
        self.batchCount = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def add_batch(self, predict, gth):
        predict = predict.flatten()
        gth = gth.flatten()

        self.hist += self.compute_hist(predict, gth)
        self.batchCount += 1

    def get_metric(self):
        epsilon = 1e-8
        overall_acc = np.diag(self.hist).sum() / (self.hist.sum() + epsilon)
        per_class_acc = np.diag(self.hist) / (self.hist.sum(1) + epsilon)
        per_class_iu = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + epsilon)
        mIOU = np.nanmean(per_class_iu)

        return overall_acc, per_class_acc, per_class_iu, mIOU
