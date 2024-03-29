from itertools import accumulate
import torch
from torchvision import transforms
import PIL
from PIL import Image
import os
import numpy as np
import pickle


im = Image.open("/data/projects/aisc_facecomp/attacks/liyi/out3000.png")
img = np.array(im)
mapping = np.bincount(img.reshape(-1))
mapping = mapping[1:]
mapping = list(reversed(mapping))
print(mapping)
out = list(accumulate(mapping))
print(out)
idx = 0
for i, v in enumerate(out):
    if v - 1254 == 1:
        idx = i + 1

zeros = np.zeros((112, 112))
ones = np.ones((112, 112))

mask = np.where(img > 255 - idx, ones, zeros)

flag = False
for i in range(111, 0, -1):
    for j in range(111, 0, -1):
        if mask[i, j] == 1 and mask[i, j + 1] == 0:
            mask[i, j] = 0
            flag = True
            break

    if flag:
        break
print(mask.sum())
print(img.shape)
mask = 1 - mask
Image.fromarray((mask * 255).astype(np.uint8)).save("out_final.png")
import pdb; pdb.set_trace()