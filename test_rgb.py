# -*- coding: utf-8 -*-
import sys, os, argparse
import PIL
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
#
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from __init__ import PACKAGE_DIR


def main():
    print '!'
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_folder', type=str, default=PACKAGE_DIR + '/dataset/', help='input image to use') # TODO: fix
    parser.add_argument('--model', type=str, default='model/best_rgb.pth', help='model file to use')
    parser.add_argument('--result_folder', type=str, default=PACKAGE_DIR + '/results/', help='where to save the output image') #TODO: fix
    opt = parser.parse_args()

    images_list = glob(os.path.join(opt.test_folder, '*.jpg'))
    print 'images_list', len(images_list)

    model = torch.load(opt.model)
    print 'Loaded'
    model = model.cuda()

    for image_path in images_list:

        img_name = image_path.split('/')[-1].split('.')[0]
        img = Image.open(image_path)

        test_img = img.resize((img.size[0]/2, img.size[1]/2), PIL.Image.BICUBIC)
        print img.size, 'after', test_img

        tensor_img = ToTensor()(test_img).unsqueeze(0)
        print tensor_img.size()

        input = Variable(tensor_img.cuda())
        HR_2, HR_4, HR_8 = model(input)

        HR_2 = HR_2.cpu()
        HR_4 = HR_4.cpu()
        HR_8 = HR_8.cpu()

        test = HR_2.data.squeeze(0)
        img = ToPILImage()(test)

        img.save(os.path.join(opt.result_folder, os.path.basename(image_path)))



if __name__ == '__main__':
    print '!'
    sys.exit(main())
