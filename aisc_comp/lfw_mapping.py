from PIL import Image
import json
import os
import shutil
import numpy as np
# with open("/data/projects/aisc_facecomp/map.txt", "r") as fp:
#     out = fp.readline()
#     out = json.loads(out)
#     print(out.keys())

# origin_image_files = [os.path.join("/data/projects/aisc_facecomp/data", "{:04d}_compare.png".format(i + 1)) for i in range(3000)]

# lfw_dir = "/data/projects/aisc_facecomp/raw_data/LFW/lfw_align_112"
# out_dir = "/data/projects/aisc_facecomp/raw_data/game3000"


# for i, file in enumerate(origin_image_files):
#     key = os.path.splitext(os.path.basename(file))[0]
#     out_class = "{:04d}_compare".format(i + 1)
#     os.makedirs(os.path.join(out_dir, out_class), exist_ok=True)
    
#     shutil.copyfile(file, os.path.join(out_dir, out_class, os.path.basename(file)))
#     class_name = out.get(key, None)
#     if class_name:
#         for cfile in os.listdir(os.path.join(lfw_dir, class_name)):
#             src_file = os.path.join(lfw_dir, class_name, cfile)
#             tgt_file = os.path.join(out_dir, out_class, cfile)
#             shutil.copyfile(src_file, tgt_file)
#     else:
#         print("without key: {}".format(key))

def check():
    dir_name = "/data/projects/aisc_facecomp/raw_data/game3000"
    imgfiles = ["{:04d}_compare.png".format(i + 1) for i in range(3000)]
    # origin_image_files = [os.path.join(dir_name, ]
    def findpic(listimg):
        for idx, img in enumerate(listimg):
            if '0001.jpg' in img:
                return idx
        raise Exception
    
    for file in imgfiles:
        class_name = os.path.splitext(file)[0]
        file1 = os.path.join(dir_name, class_name, file)
        cfiles = os.listdir(os.path.join(dir_name, class_name))
        file2 = os.path.join(dir_name, class_name, cfiles[findpic(cfiles)])
        # print(file1)
        # print(file2)
        # import pdb; pdb.set_trace()
        img1 = np.array(Image.open(file1)).astype(np.float32)
        img2 = np.array(Image.open(file2)).astype(np.float32)
        diff = img1 - img2

        # import pdb; pdb.set_trace()
        if (img1 != img2).sum() == 0:
            print("niubi: {}".format(class_name))
            # print(diff.max(), diff.min())
            # print("error in {}".format(class_name))

check()