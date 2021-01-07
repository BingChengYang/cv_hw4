import csv
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from os import listdir
from os.path import isfile, join
from random import randint

def load_img(path):
    return Image.open(path).convert('RGB')

class Load_traindata(Dataset):
    def __init__(self, transform=None, lr_transform=None, hr_transform=None, loader=load_img, scale=3, crop_size=78):
        self.dir = './training_hr_images/training_hr_images/'
        self.imgs = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.transform = transform
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.loader = load_img
        self.scale = scale
        self.crop_size = crop_size
    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(self.dir+filename)
        if self.transform is not None:
            img = self.transform(img)
            # img_hr = transforms.ToTensor()(img)
            scale = randint(2, 4)
            img_lr = transforms.Resize((self.crop_size//scale, self.crop_size//scale))(img)
            img_lr = transforms.Resize((self.crop_size, self.crop_size))(img_lr)
            img_lr = transforms.ToTensor()(img_lr)
            # img_lr = self.lr_transform(img)
            img_hr = self.hr_transform(img)

        return img_lr, img_hr

    def __len__(self):
        return len(self.imgs)

class Load_testdata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img, scale=3):
        self.dir = './testing_lr_images/testing_lr_images/'
        self.imgs = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img
        self.scale = scale
    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(self.dir+filename)
        if self.transform is not None:
            (width, height) = (3*img.width, 3*img.height)
            img = img.resize((width, height))
            img = self.transform(img)
        img_rgb = Image.open(self.dir+filename).convert('RGB')
        # img_rgb = img_rgb.resize((3*img_rgb.width, 3*img_rgb.height))
        img_ycbcr = np.array(img_rgb.convert("YCbCr")).astype(float)

        return img, img_ycbcr, filename

    def __len__(self):
        return len(self.imgs)