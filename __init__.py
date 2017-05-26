#-*- coding: utf8 -*-
import os

__all__ = ['PACKAGE_DIR', ]

PACKAGE_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')

#import torch
#print torch.has_cudnn
#print torch.cuda.is_available()
