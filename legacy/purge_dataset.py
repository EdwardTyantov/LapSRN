# -*- coding: utf-8 -*-
"Done for deleting small pictures"
import os, sys
from PIL import Image


def main():
    if len(sys.argv) > 1:
        root = sys.argv[1]
        if not os.path.exists(root):
            raise ValueError, 'No such directory %s' % (root, )
    else:
        # TODO: bsd dataset
        root = '/home/tyantov/real_photo_12k_resized'

    min_size = 256
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        img = Image.open(fpath)
        a = min(img.size)
        if a < min_size:
            print fpath, img.size
            os.remove(fpath)