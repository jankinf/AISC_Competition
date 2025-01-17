#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp

import numpy as np
from PIL import Image
import labelme
import pickle
'''
labels.txt


__ignore__
_background_
left_eye
right_eye
nose
mouth

__ignore__
_background_
mask
'''
def main():
    input_dir = "/data/projects/aisc_facecomp/finals/keypoints/tmode_masks"
    output_dir = "/data/projects/aisc_facecomp/finals/keypoints/tmode_masks"
    labels = "/data/projects/aisc_facecomp/keypoints/labels.txt"
    os.makedirs(osp.join(output_dir, "SegmentationClassPNG"), exist_ok=True)
    print("Creating dataset:", output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)


    filenames = []
    masks = []
    for filename in glob.glob(osp.join(input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        img_file = base + ".png"
        filenames.append(img_file)
        out_png_file = osp.join(
            output_dir, "SegmentationClassPNG", img_file
        )


        img = np.array(Image.open(osp.join("/data/projects/aisc_facecomp/finals/keypoints/face_aligned224", img_file)))

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        masks.append(lbl)

        labelme.utils.lblsave(out_png_file, lbl)


if __name__ == "__main__":
    main()