# -*- coding: utf-8 -*-
import os, tarfile, random
import sys
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from dataset import DatasetFromFolder
from PIL import Image


CROP_SIZE = 256


class RandomRotate(object):

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 4))


class RandomScale(object):

    def __call__(self, img):
        shape = img.size
        ratio = min(shape)/CROP_SIZE
        scale = random.uniform(ratio, 1)
        return img.resize((int(shape[0]*scale), int(shape[1]*scale)), Image.BICUBIC)


def download_bsd300(dest='dataset'):
    output_image_dir = os.path.join(dest, "BSDS300/images")

    if not os.path.exists(output_image_dir):
        if not os.path.exists(dest):
            os.makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = os.path.join(dest, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        os.remove(file_path)

    return output_image_dir


def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//8),
        ToTensor(),
    ])


def HR_2_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])


def HR_4_transform(crop_size):
    return Compose([
        Scale(crop_size//2),
        ToTensor(),
    ])


def HR_8_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomScale(),
        RandomRotate(),
        RandomHorizontalFlip(),
    ])

def get_training_set(train_dir):

    return DatasetFromFolder(train_dir,
                             LR_transform=LR_transform(CROP_SIZE),
                             HR_2_transform=HR_2_transform(CROP_SIZE),
                             HR_4_transform=HR_4_transform(CROP_SIZE),
                             HR_8_transform=HR_8_transform(CROP_SIZE))


def get_test_set():
    root_dir = download_bsd300()
    test_dir = os.path.join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             LR_transform=LR_transform(CROP_SIZE),
                             HR_2_transform=HR_2_transform(CROP_SIZE),
                             HR_4_transform=HR_4_transform(CROP_SIZE),
                             HR_8_transform=HR_8_transform(CROP_SIZE))
