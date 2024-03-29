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
SegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/comp_json_v3/SegmentationClassPNG'
SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/fill1254_comp_tmode_v3/'
# SAVESegmentationClassPNGROOT = '/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode_v2/'
os.makedirs(SAVESegmentationClassPNGROOT, exist_ok=True)

color_map = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4}
color_map_reverse = {0:(0, 0, 0), 1:(128, 0, 0), 2:(0, 128, 0), 3:(128, 128, 0), 4:(0, 0, 128)}
# 1 left eye, 2 right eye, 3 nose, 4 mouse

thres = 0
for file in os.listdir(SegmentationClassPNGROOT):
    file_path = os.path.join(SegmentationClassPNGROOT, file)
    compare_image = Image.open(file_path).convert('RGB')
    compare_image = np.array(compare_image)
    (rows, cols) = compare_image.shape[:2]

    tmp_comp_mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            tmp_comp_mask[i, j] = color_map[tuple(compare_image[i, j])]

    nonzero_pixels = np.sum(tmp_comp_mask != 0).item()
    compare_image = np.stack([tmp_comp_mask] * 3, axis=2)

    print(nonzero_pixels)
    fill_sign = nonzero_pixels < 1254
    # 增大像素个数
    if fill_sign:
        add_pixels = 1254 - nonzero_pixels
        count_pixels = 0
        pre = 0
        loop_cnt = 0

        while count_pixels < add_pixels:
            mask = np.copy(compare_image)

            mask_left_eye = np.where(compare_image == 1, 1, 255)
            mask_right_eye = np.where(compare_image == 2, 2, 255)
            mask_mouse = np.where(compare_image == 4, 4, 255)
            mask_nose = np.where(compare_image == 3, 3, 255)

            def func_add(val, matrix):
                global count_pixels
                global add_pixels
                
                for i in range(1, rows - 1):
                    if count_pixels == add_pixels:
                        break
                    for j in range(1, cols - 1):
                        if count_pixels == add_pixels:
                            break
                        cur_val = matrix[i, j].tolist()
                        up_val = matrix[i - 1, j].tolist()
                        down_val = matrix[i + 1, j].tolist()
                        left_val = matrix[i, j - 1].tolist()
                        right_val = matrix[i, j + 1].tolist()
                        if cur_val == [val] * 3:
                            continue
                        if cur_val == up_val == down_val == left_val == right_val:
                            continue
                        if mask[i, j].tolist() != [0] * 3:
                            continue
                        else:
                            mask[i, j] = val
                        count_pixels += 1

            func_add(val=3, matrix=mask_nose)
            diff = count_pixels - pre
            pre = count_pixels
            print("add_nose:", diff)

            if loop_cnt > thres:
                func_add(val=1, matrix=mask_left_eye)
                diff = count_pixels - pre
                pre = count_pixels
                print("add_left_eye:", diff)

                func_add(val=2, matrix=mask_right_eye)
                diff = count_pixels - pre
                pre = count_pixels
                print("add_right_eye:", diff)

                func_add(val=4, matrix=mask_mouse)
                diff = count_pixels - pre
                pre = count_pixels
                print("add_mouse:", diff)

            loop_cnt += 1

            for i in range(rows):
                for j in range(cols):
                    if mask[i, j].tolist() == [0] * 3:
                        continue
                    compare_image[i, j] = mask[i, j]

    # 减小像素个数
    else:
        remove_pixels = nonzero_pixels - 1254
        count_pixels = 0
        pre = 0
        loop_cnt = 0
        while count_pixels < remove_pixels:
            mask = np.ones((112, 112, 3))

            mask_left_eye = np.where(compare_image == 1, 1, 255)
            mask_right_eye = np.where(compare_image == 2, 2, 255)
            mask_nose = np.where(compare_image == 3, 3, 255)
            mask_mouse = np.where(compare_image == 4, 4, 255)

            def func_sub(matrix):
                global count_pixels
                global remove_pixels
                
                for i in range(1, rows - 1):
                    if count_pixels == remove_pixels:
                        break
                    for j in range(1, cols - 1):
                        if count_pixels == remove_pixels:
                            break
                        cur_val = matrix[i, j].tolist()
                        up_val = matrix[i - 1, j].tolist()
                        down_val = matrix[i + 1, j].tolist()
                        left_val = matrix[i, j - 1].tolist()
                        right_val = matrix[i, j + 1].tolist()
                        if cur_val == [255] * 3:
                            continue
                        if cur_val == up_val == down_val == left_val == right_val:
                            continue
                        mask[i, j] = 0
                        count_pixels += 1

            func_sub(matrix=mask_mouse)
            diff = count_pixels - pre
            pre = count_pixels
            print("sub_mouse:", diff)

            if loop_cnt > thres:
                func_sub(matrix=mask_nose)
                diff = count_pixels - pre
                pre = count_pixels
                print("sub_nose:", diff)
                
                func_sub(matrix=mask_left_eye)
                diff = count_pixels - pre
                pre = count_pixels
                print("sub_left_eye:", diff)
                
                func_sub(matrix=mask_right_eye)
                diff = count_pixels - pre
                pre = count_pixels
                print("sub_right_eye:", diff)
            
            loop_cnt += 1
            
            for i in range(rows):
                for j in range(cols):
                    if mask[i, j].tolist() == [0] * 3:
                        compare_image[i, j] = 0

    valid_mask = np.where(compare_image.sum(axis=2) != 0, 255, 0)
    compare_image = np.stack([valid_mask] * 3, axis=2)
    
    assert np.sum(compare_image == 255) == (1254 * 3)

    compare_image = 255 - compare_image
    masked_img = Image.fromarray(compare_image.astype(np.uint8))
    masked_img.save(os.path.join(SAVESegmentationClassPNGROOT, file))
