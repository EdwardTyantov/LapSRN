# -*- coding: utf-8 -*-
import os, random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, LR_transform=None, HR_2_transform=None,  HR_4_transform=None, HR_8_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]

        self.LR_transform = LR_transform
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform
        self.HR_8_transform = HR_8_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        HR_8 = self.HR_8_transform(input)
        HR_4 = self.HR_4_transform(HR_8)
        HR_2 = self.HR_2_transform(HR_8)
        LR = self.LR_transform(HR_8)

        to_tensor = torchvision.transforms.ToTensor()
        HR_8 = to_tensor(HR_8)
        return LR, HR_2, HR_4, HR_8

    def __len__(self):
        return len(self.image_filenames)
