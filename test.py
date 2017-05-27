# -*- coding: utf-8 -*-
import sys, os, argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from __init__ import TEST_DIR


def centeredCrop(img):
    width, height = img.size  # Get dimensions
    new_width = width - width % 8
    new_height = height - height % 8
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


def process(out, cb, cr):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img


def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_folder', type=str, default=TEST_DIR, help='input image to use')
    parser.add_argument('--model', type=str, default='model/model_epoch_1.pth', help='model file to use')
    parser.add_argument('--save_folfer', type=str, default='./results', help='input image to use')
    parser.add_argument('--output_filename', type=str, help='where to save the output image')
    opt = parser.parse_args()

    images_list = glob(os.path.join(opt.test_folder, '*.jpg'))
    print 'images_list', len(images_list)

    model = torch.load(opt.model)
    print opt.model
    print 'Loaded'
    model = model.cuda()

    for image_path in images_list:
        print image_path
        img_name = image_path.split('/')[-1].split('.')[0]
        img = Image.open(image_path).convert('YCbCr')
        img = centeredCrop(img)
        y, cb, cr = img.split()
        # TODO: bug with y resize
        LR = y  # .resize((y.size[0]/8, y.size[1]/8), Image.BICUBIC)
        print LR.size
        LR = Variable(ToTensor()(LR)).view(1, -1, LR.size[1], LR.size[0])
        LR = LR.cuda()
        HR_2, HR_4, HR_8 = model(LR)
        HR_2 = HR_2.cpu()
        HR_4 = HR_4.cpu()
        HR_8 = HR_8.cpu()

        HR_2 = process(HR_2, cb, cr)
        HR_4 = process(HR_4, cb, cr)
        HR_8 = process(HR_8, cb, cr)
        HR_2.save(opt.save_folfer + '/' + "%s-mul2.jpg" % img_name)
        HR_4.save(opt.save_folfer + '/' + "%s-mul4.jpg" % img_name)
        HR_8.save(opt.save_folfer + '/' + "%s-mul8.jpg" % img_name)
        print 'Done'
        break


if __name__ == '__main__':
    sys.exit(main())