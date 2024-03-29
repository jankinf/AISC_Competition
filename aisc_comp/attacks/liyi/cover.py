import torch
from torchvision import transforms
import PIL
from PIL import Image
import os
import numpy as np
import pickle

# SegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/comp_json/SegmentationClassPNG'
# SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_v2/'
# SegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/origin_json/SegmentationClassPNG'
# SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_origin_v2/'
# SegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/comp_json_v2/SegmentationClassPNG'
# SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode/'
SegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/comp_json_v2/SegmentationClassPNG'
SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/cover/'
# SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode_v2/'
os.makedirs(SAVESegmentationClassPNGROOT, exist_ok=True)

color_map = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4}
color_map_reverse = {0:(0, 0, 0), 1:(128, 0, 0), 2:(0, 128, 0), 3:(128, 128, 0), 4:(0, 0, 128)}
# 1 left eye, 2 right eye, 3 nose, 4 mouse

thres = 0
files = os.listdir(SegmentationClassPNGROOT)
tmp_comp_masks = []
for file in files:
    file_path = os.path.join(SegmentationClassPNGROOT, file)
    compare_image = Image.open(file_path).convert('RGB')
    compare_image = np.array(compare_image)
    (rows, cols) = compare_image.shape[:2]

    tmp_comp_mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            tmp_comp_mask[i, j] = color_map[tuple(compare_image[i, j])]
    tmp_comp_masks.append(tmp_comp_mask)
    

# zeros = np.zeros((112, 112))
# ones = np.ones((112, 112))
# allmask = np.zeros((112, 112))
# for i in range(1, 5):
#     mask = np.zeros((112, 112))
#     for tmp_comp_mask in tmp_comp_masks:
#         mask += np.where(tmp_comp_mask == i, ones, zeros)
    
#     maxn = np.max(mask)
#     mask /= maxn

#     allmask += mask
# allmask = np.clip(allmask, 0, 1)
# Image.fromarray((allmask * 255).astype(np.uint8)).save("out.png")



zeros = np.zeros((112, 112))
ones = np.ones((112, 112))
allmask = np.zeros((112, 112))
freq = 20
for idx, tmp_comp_mask in enumerate(tmp_comp_masks):
    mask = np.zeros((112, 112))
    for i in range(1, 5):
        mask += np.where(tmp_comp_mask == i, ones, zeros)
    
    allmask += mask
    if (idx + 1) % freq == 0:
        maxn = np.max(allmask)
        Image.fromarray((allmask / maxn * 255).astype(np.uint8)).save("out{}.png".format(idx + 1))
# import pdb; pdb.set_trace()
# Image.fromarray((mask * 255).astype(np.uint8)).save("out{}.png".format(i))
