# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import os.path as osp

class COVIDDataset(data.Dataset):
    def __init__(self, data_dir, trainsize, crossVal):
        self.trainsize = trainsize
        self.data_dir = data_dir
        self.images = []
        self.gts = []
        self.edges = []
        fileName = 'COVID-19-CT100/dataList/'+'trainEdge'+str(crossVal)+'.txt'
        with open(osp.join(self.data_dir, fileName), 'r') as text_file:
            for line in text_file:
                line_arr = line.split()
                img_file = osp.join(self.data_dir, line_arr[0].strip())
                label_file = osp.join(self.data_dir, line_arr[1].strip())
                edge_file = osp.join(self.data_dir, line_arr[2].strip())
                self.images.append(img_file)
                self.gts.append(label_file)
                self.edges.append(edge_file)


            '''
            self.images = [self.data_dir + line.split()[0].strip() for line in text_file]
            self.gts = [self.data_dir + line.split()[1].strip() for line in text_file]
            self.edges = [self.data_dir + line.split()[2].strip() for line in text_file]
            '''

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        self.edge_flage = True

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt

    def filter_files(self):
        #print(len(self.images), len(self.gts))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(data_dir, crossVal, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = COVIDDataset(data_dir, trainsize, crossVal)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader


class test_dataset:
    def __init__(self, data_dir, testsize):
        self.testsize = testsize
        #self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        fileName = 'COVID-19-' + dataset + '/dataList/'+'trainEdge'+str(crossVal)+'.txt'
        with open(osp.join(self.data_dir, fileName), 'r') as text_file:
            self.images = [self.data_dir + line.split()[0].strip() for line in text_file]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        image = self.transform(image).unsqueeze(0)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
