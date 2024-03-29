import os
import numpy as np
import pandas as pd
import shutil
from collections import Counter


df = None

def get_indice(file):
    global df
    df_tmp = pd.read_csv(file)
    name = os.path.splitext(os.path.basename(file))[0]
    name = name[:name.rfind('@')]
    
    if df is None:
        df = df_tmp.copy()
    else:
        indice = (df_tmp.iloc[:, 0].values > df.iloc[:, 0].values).nonzero()[0]
        df.iloc[indice, :] = df_tmp.iloc[indice, :]
 

def generate_data(csv, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    dir_names = pd.read_csv(csv).iloc[:, -1].values
    filenames = ["{:04d}.png".format(i + 1) for i in range(3000)]

    for dir_name, filename in zip(dir_names, filenames):
        src_file = os.path.join("/data/projects/aisc_facecomp/results", dir_name, filename)
        tgt_file = os.path.join(to_dir, filename)
        shutil.copyfile(src_file, tgt_file)

files = [
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model0134578.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_cmp_mask_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ori_mask_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model0134781315.csv",
    "/data/projects/verification/aisc_csv/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ord2_model01347813.csv",
    "/data/projects/verification/aisc_csv/origin_mask_merge.csv",
]

for file in files:
    get_indice(file)

all_dirs = []
for path in df.iloc[:, -1]:
    all_dirs.append(os.path.dirname(path))
print(Counter(all_dirs))
import pdb; pdb.set_trace()
save_name = "final"
df.to_csv("cos_stat/repo/{}.csv".format(save_name), index=False)

generate_data("cos_stat/repo/{}.csv".format(save_name), "/data/projects/aisc_facecomp/results/final")