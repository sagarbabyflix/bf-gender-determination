import albumentations as A
import cv2
import random
import numpy as np
import os.path as osp
import torch

from torch.utils import data


NONETYPE = type(None)


class ImageDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 flip=False,
                 verbose=True,
                 test_mode=False,
                 return_name=False,
                 return_imsize=False,
                 **kwargs):
        self.inputs = inputs
        self.labels = labels
        self.resize = resize
        self.augment = augment
        self.crop = crop 
        self.preprocess = preprocess
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_imsize = return_imsize

    def __len__(self): return len(self.inputs)

    def process_image(self, X):
        if self.resize: X = self.resize(image=X)['image']
        if self.augment: X = self.augment(image=X)['image']
        if self.crop: X = self.crop(image=X)['image']
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
        X = np.ascontiguousarray(X)
        return X

    def get(self, i):
        try:
            X = cv2.imread(self.inputs[i])
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        imsize = X.shape[:2]
        X = self.process_image(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i])

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)

