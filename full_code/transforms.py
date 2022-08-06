import numpy as np
import torch
import random
import cv2

class Scale(object):
    """
    Resize the given image to a fixed scale
    """
    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return [img, label]

class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]

class RandomFlip(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """
    def __call__(self, image, label):
        if random.random() < 0.5:
            x1 = 0#random.randint(0, 1) # if you want to do vertical flip, uncomment this line
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1) # veritcal flip
        return [image, label]

class RandomScale(object):
    """
    Randomly resize the given image
    """
    def __init__(self, min_scale=0.5, max_scale=2.0):
        '''
        :param min_scale: minimum scale of resizing
        :param max_scale: maximum scale of resizing
        '''
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # select a random scale
        scale = self.min_scale + random.randint(0, int((self.max_scale - self.min_scale)*10)) / 10.0
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        return [img, label]

class RandomCrop(object):
    """
    Randomly crop the given image
    """
    def __init__(self, crop_size, ignore_index=255):
        '''
        :param crop_size: size after cropping
        '''
        if isinstance(crop_size, int):
            self.crop_h = crop_size
            self.crop_w = crop_size
        elif isinstance(crop_size, list):
            self.crop_h = crop_size[0]
            self.crop_w = crop_size[1]
        else:
            ValueError('Unknown crop_size format')
        self.ignore_index = ignore_index

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: cropped images
        '''
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(self.ignore_index,))
        else:
            img_pad, label_pad = img, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        img = img_pad[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label_pad[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]

        return [img, label]

class Normalize(object):
    """
    Given mean: (B, G, R) and std: (B, G, R),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        image = ((image / 255 - self.mean) / self.std).astype(np.float32)
        label = label / 255
        '''
        for i in range(3):
            image[:, :, i] -= self.mean[i]
        for i in range(3):
            image[:, :, i] /= self.std[i]
        '''
        return [image, label]


class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __init__(self, scale=1):
        '''
        :param scale: set this parameter according to the output scale
        '''
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), \
                interpolation=cv2.INTER_NEAREST)
        image = image.transpose((2, 0, 1))

        image_tensor = torch.from_numpy(image)#.div(255)
        label_tensor =  torch.LongTensor(np.array(label, dtype=np.int))

        return [image_tensor, label_tensor]

class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
