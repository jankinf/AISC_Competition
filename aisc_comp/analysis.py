import os
import numpy as np
import pandas as pd

IMG_DIR="/data/projects/aisc_facecomp/data"

def get_indice(file):
    df = pd.read_csv(file)
    ary = df.iloc[:, -1].values
    med = np.median(ary)
    indice = (ary < med).nonzero()[0]
    return indice

def get_mean_score(file):
    df = pd.read_csv(file)
    score = (df.iloc[:1000] > 0.22).mean(0)[-1]
    return score

# print(indice.shape)

# files = [os.path.join(IMG_DIR, "{:04d}.png".format(i + 1)) for i in range(3000)]
# import pdb; pdb.set_trace()
# for idx in indice:
#     print(files[idx])
# self.compare_image_files = [os.path.join(self.IMG_DIR, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]

# file1 = "/data/projects/aisc_facecomp/cos_stat/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178@3:0.22-4:0.22-5:0.22.csv"
# idx1 = get_indice(file1)

# file2 = "/data/projects/aisc_facecomp/cos_stat/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178@3:0.22-4:0.22-5:0.22.csv"
# idx2 = get_indice(file2)
# idx1 = set(idx1)
# idx2 = set(idx2)
# int_idx = idx1.intersection(idx2)

allidx = set([i for i in range(3000)])
for filename in os.listdir('cos_stat'):
    file = os.path.join('cos_stat', filename)
    idx = set(get_indice(file))
    allidx = allidx.intersection(idx)
    print(file)
    print(get_mean_score(file))

print(len(allidx))


import pdb; pdb.set_trace()