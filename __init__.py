#-*- coding: utf8 -*-
import os

__all__ = ['PACKAGE_DIR', 'DATASET_DIR', 'BSD_URL', 'TEST_DIR']

PACKAGE_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
DATASET_DIR = os.path.join(PACKAGE_DIR, 'dataset')
TEST_DIR = os.path.join(PACKAGE_DIR, 'test_dataset')
BSD_URL = 'http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz'

#import torch
#print torch.has_cudnn
#print torch.cuda.is_available()
