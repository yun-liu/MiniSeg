import numpy as np
import cv2
from PIL import Image
import pickle
import os.path as osp


class LoadData(object):
    def __init__(self, data_dir, classes, normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))


    def readFile(self, fileName, trainStg=False):
        if trainStg == True:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        with open(osp.join(self.data_dir, fileName), 'r') as text_file:
            for line in text_file:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split()
                img_file = osp.join(self.data_dir, line_arr[0].strip())
                label_file = osp.join(self.data_dir, line_arr[1].strip())
                label_img = cv2.imread(label_file, 0) / 255

                if trainStg == True:
                    hist = np.histogram(label_img, self.classes, range=(0, self.classes))
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file) / 255#BGR
                    rgb_img = rgb_img[:, :, ::-1]#RGB
                    self.mean[0] += np.mean(rgb_img[:,:,0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)
                no_files += 1

        if trainStg == True:
            # divide the mean and std values by the sample space size
            self.mean /= no_files
            self.std /= no_files

            #compute the class imbalance information
            self.compute_class_weights(global_hist)
        return 0

    def processData(self, crossVal, dataset):
        print('Processing training data')
        return_train = self.readFile('COVID-19-' + dataset + '/dataList/'+'train'+str(crossVal)+'.txt', True)

        print('Processing validation data')
        return_val = self.readFile('COVID-19-' + dataset + '/dataList/'+'val'+str(crossVal)+'.txt')

        print('Pickling data')
        if return_train == 0 and return_val == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights

            return data_dict
        return None
