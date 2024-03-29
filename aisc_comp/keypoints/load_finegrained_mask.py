#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp

import numpy as np
from PIL import Image
import labelme

input_dir = "/data/projects/aisc_facecomp/keypoints/comp_json/SegmentationClassPNG"
output_dir = "/data/projects/aisc_facecomp/keypoints/comp_json/masked_img"
# src_dir = "/data/projects/aisc_facecomp/keypoints/comp_json"
src_dir = "/data/projects/aisc_facecomp/data"

def main():
    mask_sizes = []
    cnt = 0
    for filename in glob.glob(osp.join(input_dir, "*.png")):
        print("Generating dataset from:", filename)

        base = osp.basename(filename)
        maskfile = osp.join(input_dir, base)
        outfile = osp.join(output_dir, base)
        srcfile = osp.join(src_dir, base)

        img = np.array(Image.open(maskfile))
        mask_sizes.append(img.sum())
        cnt += img.sum()
        mask = 1 - img.astype(np.float32)
        mask = np.stack([mask] * 3, axis=2)
        # print(mask.shape)
        # import pdb; pdb.set_trace()

        src = np.array(Image.open(srcfile)).astype(np.float32)
        masked_img = src * mask
        Image.fromarray(masked_img.astype(np.uint8)).save(outfile)

    for msize in mask_sizes:
        print(msize)
    print("mean: {}".format(cnt / len(mask_sizes)))

if __name__ == "__main__":
    main()