import os, sys
from PIL import Image


root = '/home/tyantov/real_photo_12k_resized'


for fname in os.listdir(root):
    fpath = os.path.join(root, fname)
    img = Image.open(fpath)
    a = min(img.size)
    if a < 256:
        print fpath, img.size